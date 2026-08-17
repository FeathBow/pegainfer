//! KV-backed serving forward: two KV families, prefill and decode, and
//! atomic dual-pool admission.
//!
//! The layer forwards take the step's plans and prep metadata and the
//! pools this module owns, never a request's state. Both coordinate systems
//! (absolute positions for RoPE, cache-relative slots for the paged
//! scatter) coincide below the
//! sliding window; past it the local family releases its front, and
//! `origin_pages` is what converts between the two.
//!
//! Decode runs the prefill entries with seq_len 1 — correct, not
//! decode-optimal. Attention reads are read-only entries (the prep kernels own the
//! pool writes) with sm_scale 1.0 (Gemma 4 runs unscaled attention) and
//! window_left already passed through.

use anyhow::Context as AnyhowContext;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::LayerKind;
use crate::forward::embed_scale_bf16;
use crate::forward::logits_tail;
use crate::forward::logits_tail_into;
use crate::forward::validate_tokens;
use crate::kv::GemmaKv;
use crate::kv::PAGE_SIZE;
use crate::kv::SlidingLocalKv;
use crate::layer::EpilogueScratch;
use crate::layer::LayerGeometry;
use crate::layer::attention_epilogue_into;
use crate::layer::build_proportional_rope_tables;
use crate::weights::Gemma4Layer;
use crate::weights::Gemma4Weights;

/// Which logits the caller wants materialized from a step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LogitsSpan {
    LastRow,
    All,
}

/// How a step feeds the preps. The tower above them is the same either way;
/// only the metadata a prep reads changes.
#[derive(Clone, Copy)]
enum PrepRef<'a> {
    Single {
        start_pos: usize,
        /// Absolute page the local resident row starts at. The preps shift
        /// only the row index by it; RoPE stays on absolute positions.
        local_page_origin: usize,
    },
    /// One row per request. Each row's page window and absolute position are
    /// the ones its family's plan already carries; what the plans do not
    /// carry is the sliding family's released front, one page index per row.
    Batched { local_origins: &'a CudaSlice<i32> },
}

struct GemmaStepPlan {
    start_pos: usize,
    local_page_origin: usize,
    local_plan: PrefillPagedPlan,
    global_plan: PrefillPagedPlan,
}

fn upload_prefix<T: cudarc::driver::DeviceRepr>(
    ctx: &DeviceContext,
    slot: &mut CudaSlice<T>,
    values: &[T],
) -> Result<()> {
    anyhow::ensure!(
        values.len() <= slot.len(),
        "step metadata of {} entries overruns its {} entry slot",
        values.len(),
        slot.len()
    );
    let mut view = slot.slice_mut(..values.len());
    ctx.stream
        .memcpy_htod(values, &mut view)
        .map_err(|err| anyhow::anyhow!("step metadata H2D failed: {err}"))
}

/// The attention path's working set. Both families run the same tower over
/// the same rows, so one set serves both: each buffer is allocated at the
/// wider family's width and reshaped per layer.
struct AttnScratch {
    normed_x: HiddenStates,
    q_states: HiddenStates,
    k_states: HiddenStates,
    v_states: HiddenStates,
    q_prep: HiddenStates,
    attn: HiddenStates,
}

impl AttnScratch {
    fn new(
        ctx: &DeviceContext,
        local: &LayerGeometry,
        global: &LayerGeometry,
        max_rows: usize,
    ) -> Result<Self> {
        let q_dim = |geom: &LayerGeometry| geom.num_q_heads * geom.head_dim;
        let kv_dim = |geom: &LayerGeometry| geom.num_kv_heads * geom.head_dim;
        let q_max = q_dim(local).max(q_dim(global));
        let kv_max = kv_dim(local).max(kv_dim(global));
        Ok(Self {
            normed_x: HiddenStates::zeros(ctx, local.hidden_size, max_rows)?,
            q_states: HiddenStates::zeros(ctx, q_max, max_rows)?,
            k_states: HiddenStates::zeros(ctx, kv_max, max_rows)?,
            v_states: HiddenStates::zeros(ctx, kv_max, max_rows)?,
            q_prep: HiddenStates::zeros(ctx, q_max, max_rows)?,
            attn: HiddenStates::zeros(ctx, q_max, max_rows)?,
        })
    }

    fn set(&mut self, geom: &LayerGeometry, seq_len: usize) {
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        for (buf, hidden_dim) in [
            (&mut self.normed_x, geom.hidden_size),
            (&mut self.q_states, q_dim),
            (&mut self.k_states, kv_dim),
            (&mut self.v_states, kv_dim),
            (&mut self.q_prep, q_dim),
            (&mut self.attn, q_dim),
        ] {
            buf.hidden_dim = hidden_dim;
            buf.seq_len = seq_len;
        }
    }
}

/// The tower's whole working set for one step: attention buffers, epilogue
/// buffers, and the hidden pair the layers alternate between so no layer
/// writes the buffer it is reading.
struct TowerScratch {
    attn: AttnScratch,
    epilogue: EpilogueScratch,
    hidden: [HiddenStates; 2],
}

impl TowerScratch {
    fn new(
        ctx: &DeviceContext,
        local: &LayerGeometry,
        global: &LayerGeometry,
        max_rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            attn: AttnScratch::new(ctx, local, global, max_rows)?,
            epilogue: EpilogueScratch::new(ctx, local, max_rows)?,
            hidden: [
                HiddenStates::zeros(ctx, local.hidden_size, max_rows)?,
                HiddenStates::zeros(ctx, local.hidden_size, max_rows)?,
            ],
        })
    }

    fn open(&mut self, seq_len: usize) -> Result<()> {
        self.epilogue.set_rows(seq_len)?;
        for buf in &mut self.hidden {
            buf.seq_len = seq_len;
        }
        Ok(())
    }
}

fn hidden_pair(hidden: &mut [HiddenStates; 2], src: usize) -> (&HiddenStates, &mut HiddenStates) {
    let (first, second) = hidden.split_at_mut(1);
    if src == 0 {
        (&first[0], &mut second[0])
    } else {
        (&second[0], &mut first[0])
    }
}

pub(crate) struct StepArena {
    tower: TowerScratch,
    local_plan: PrefillPagedPlan,
    global_plan: PrefillPagedPlan,
    local_origins: CudaSlice<i32>,
    ids: CudaSlice<u32>,
    head_normed: HiddenStates,
    logits: HiddenStates,
    max_rows: usize,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

impl StepArena {
    /// Settle the step's preconditions — the allocation stream and a row
    /// count the buffers can hold — since neither can be safely deferred
    /// until the epilogue, by which point the step has already written KV.
    fn open(&mut self, ctx: &DeviceContext, rows: usize) -> Result<()> {
        anyhow::ensure!(
            std::sync::Arc::ptr_eq(&ctx.stream, &self.stream),
            "a step arena must be used on the stream it was allocated on"
        );
        anyhow::ensure!(
            rows <= self.max_rows,
            "step of {rows} rows exceeds the arena's {} row ceiling",
            self.max_rows
        );
        self.tower.open(rows)
    }
}

/// Everything a serving step needs that outlives requests.
pub(crate) struct GemmaServe {
    /// The weights these pools, rope tables and layer numbering were built
    /// for. Holding them is what makes a step's model identity structural:
    /// KV pages written under one checkpoint are meaningless under another.
    weights: Gemma4Weights,
    pub(crate) local_pool: KvPool,
    pub(crate) global_pool: KvPool,
    local_geom: LayerGeometry,
    global_geom: LayerGeometry,
    sliding_window: usize,
    final_logit_softcapping: f32,
    #[cfg(test)]
    release_enabled: bool,
    sliding_cos: DeviceVec,
    sliding_sin: DeviceVec,
    global_cos: DeviceVec,
    global_sin: DeviceVec,
    cos_max_pos: usize,
    /// Model layer index -> index within its family's pool layer axis.
    family_index: Vec<usize>,
}

impl GemmaServe {
    pub(crate) fn new(
        ctx: &DeviceContext,
        weights: Gemma4Weights,
        max_context: usize,
        local_pages: usize,
        global_pages: usize,
    ) -> Result<Self> {
        // One source of truth for geometry, rope tables and layer numbering.
        let config = &weights.config;
        let device_ordinal = ctx.device_ordinal;
        // One tensor answers for the set: the loader materializes every
        // weight through a single context.
        let weights_ordinal = weights.embed_tokens.data.ordinal();
        anyhow::ensure!(
            weights_ordinal == device_ordinal,
            "weights live on device {weights_ordinal} but this context \
             allocates on device {device_ordinal}"
        );
        let (mut locals, mut globals) = (0usize, 0usize);
        let family_index = config
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerKind::Sliding => {
                    locals += 1;
                    locals - 1
                }
                LayerKind::Global => {
                    globals += 1;
                    globals - 1
                }
            })
            .collect();
        let local_pool = KvPool::new(
            ctx,
            locals,
            config.num_key_value_heads,
            config.head_dim,
            PAGE_SIZE,
            local_pages,
        )?;
        let global_pool = KvPool::new(
            ctx,
            globals,
            config.num_global_key_value_heads,
            config.global_head_dim,
            PAGE_SIZE,
            global_pages,
        )?;
        let local_geom = LayerGeometry::local_of(config);
        let global_geom = LayerGeometry::global_of(config);
        let (sliding_cos, sliding_sin) = pegainfer_core::rope::precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: local_geom.head_dim,
                frequency_dim: local_geom.head_dim,
                max_seq_len: max_context,
                theta: config.sliding_rope_theta,
            },
        )?;
        let (global_cos, global_sin) = build_proportional_rope_tables(
            ctx,
            config.global_rope_theta,
            global_geom.head_dim,
            config.global_rotary_dim,
            max_context,
        )?;
        let (sliding_window, final_logit_softcapping) =
            (config.sliding_window, config.final_logit_softcapping);
        Ok(Self {
            weights,
            local_pool,
            global_pool,
            local_geom,
            global_geom,
            sliding_window,
            final_logit_softcapping,
            #[cfg(test)]
            release_enabled: true,
            sliding_cos,
            sliding_sin,
            global_cos,
            global_sin,
            cos_max_pos: max_context,
            family_index,
        })
    }

    /// One arena per engine thread, sized for the decode step; a prompt
    /// builds a [`TowerScratch`] for its own width instead. A request's
    /// tiles are `ceil(rows * group / cta_tile_q)` with a positive
    /// `cta_tile_q`, so `rows * group` bounds each plan.
    pub(crate) fn alloc_step_arena(
        &self,
        ctx: &DeviceContext,
        max_rows: usize,
    ) -> Result<StepArena> {
        let group = |geom: &LayerGeometry| geom.num_q_heads / geom.num_kv_heads;
        let alloc = |err: &'static str| {
            move |e: cudarc::driver::DriverError| anyhow::anyhow!("{err} alloc failed: {e}")
        };
        Ok(StepArena {
            tower: TowerScratch::new(ctx, &self.local_geom, &self.global_geom, max_rows)?,
            local_plan: PrefillPagedPlan::new_preallocated(
                ctx,
                max_rows,
                self.local_pool.capacity_pages(),
                max_rows,
                max_rows * group(&self.local_geom),
            )?,
            global_plan: PrefillPagedPlan::new_preallocated(
                ctx,
                max_rows,
                self.global_pool.capacity_pages(),
                max_rows,
                max_rows * group(&self.global_geom),
            )?,
            local_origins: ctx.stream.alloc_zeros(max_rows).map_err(alloc("origins"))?,
            ids: ctx.stream.alloc_zeros(max_rows).map_err(alloc("ids"))?,
            head_normed: HiddenStates::zeros(ctx, self.local_geom.hidden_size, max_rows)?,
            logits: HiddenStates::zeros(ctx, self.weights.embed_tokens.rows, max_rows)?,
            max_rows,
            stream: ctx.stream.clone(),
        })
    }

    pub(crate) fn alloc_kv(&self) -> GemmaKv {
        GemmaKv {
            local: SlidingLocalKv::new(self.local_pool.clone()),
            global: self.global_pool.alloc(),
        }
    }

    /// The eviction gate runs the same request twice, once with the front
    /// held resident, to show what release does and does not change.
    #[cfg(test)]
    pub(crate) fn set_release_for_test(&mut self, on: bool) {
        self.release_enabled = on;
    }

    fn check_step_bounds(&self, kv: &GemmaKv, kv_len: usize) -> Result<()> {
        anyhow::ensure!(
            kv.local.seq_len() == kv.global.seq_len(),
            "the two families' frontiers diverged: local {} global {}",
            kv.local.seq_len(),
            kv.global.seq_len()
        );
        anyhow::ensure!(
            kv.local.belongs_to(&self.local_pool) && kv.global.belongs_to(&self.global_pool),
            "a KV state came from another pool; its page ids do not address \
             this one's buffer"
        );
        anyhow::ensure!(
            kv_len <= self.cos_max_pos,
            "kv_len {kv_len} exceeds rope tables' {} rows",
            self.cos_max_pos
        );
        Ok(())
    }

    fn plan_step(
        &self,
        ctx: &DeviceContext,
        kv: &GemmaKv,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<GemmaStepPlan> {
        let kv_len = start_pos + seq_len;
        let global_desc = kv.global.desc_for_len(kv_len)?;
        // The local plan lives in cache-relative coordinates: the resident row
        // starts `origin_pages` pages into the sequence, and window_left masks
        // whatever sub-window prefix the first page still carries, so a
        // resident start that is not window-aligned loses nothing.
        let page = kv.local.layout().page_size;
        let origin_tokens = kv.local.origin_pages() * page;
        let rel_kv_len = kv_len
            .checked_sub(origin_tokens)
            .context("the resident window starts past the step's frontier")?;
        let rel_start = start_pos
            .checked_sub(origin_tokens)
            .context("the step starts before the resident window")?;
        let row = kv.local.page_row();
        anyhow::ensure!(
            row.len() == rel_kv_len.div_ceil(page),
            "local resident row of {} pages against {rel_kv_len} tokens",
            row.len()
        );
        let rel_last_page = if rel_kv_len.is_multiple_of(page) {
            page
        } else {
            rel_kv_len % page
        };
        let local_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            ctx,
            &[row],
            &[rel_last_page],
            &[rel_start],
            &[seq_len],
            self.local_geom.num_q_heads,
            self.local_geom.num_kv_heads,
            self.local_geom.head_dim,
            0,
        )?;
        let global_plan = PrefillPagedPlan::new(
            ctx,
            &global_desc,
            start_pos,
            seq_len,
            self.global_geom.num_q_heads,
            self.global_geom.num_kv_heads,
            self.global_geom.head_dim,
        )?;
        Ok(GemmaStepPlan {
            start_pos,
            local_page_origin: kv.local.origin_pages(),
            local_plan,
            global_plan,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn local_layer_serve(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        layer: &Gemma4Layer,
        family_layer: usize,
        seq_len: usize,
        prep: PrepRef<'_>,
        local_plan: &PrefillPagedPlan,
        global_plan: &PrefillPagedPlan,
        src: usize,
    ) -> Result<()> {
        let geom = &self.local_geom;
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        let v_proj = layer
            .attention
            .v_proj
            .as_ref()
            .context("local layer requires v_proj")?;
        let TowerScratch {
            attn: scratch,
            epilogue,
            hidden,
        } = tower;
        scratch.set(geom, seq_len);
        let (x, out) = hidden_pair(hidden, src);

        ops::rms_norm_batch_into(
            ctx,
            x,
            &layer.input_layernorm,
            geom.rms_norm_eps,
            &mut scratch.normed_x,
        );
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.q_proj,
            0,
            q_dim,
            &scratch.normed_x,
            &mut scratch.q_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.k_proj,
            0,
            kv_dim,
            &scratch.normed_x,
            &mut scratch.k_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            v_proj,
            0,
            kv_dim,
            &scratch.normed_x,
            &mut scratch.v_states,
        )?;

        match prep {
            PrepRef::Single {
                start_pos,
                local_page_origin,
            } => {
                ops::qkv_norm_rope_paged_prefill_hd256_plain_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &scratch.v_states,
                    &mut scratch.q_prep,
                    self.local_pool.buffer(),
                    &self.local_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.sliding_cos,
                    &self.sliding_sin,
                    family_layer,
                    local_plan.page_indices_d(),
                    local_page_origin,
                    start_pos,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
            PrepRef::Batched { local_origins } => {
                ops::qkv_norm_rope_paged_decode_hd256_plain_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &scratch.v_states,
                    &mut scratch.q_prep,
                    self.local_pool.buffer(),
                    &self.local_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.sliding_cos,
                    &self.sliding_sin,
                    family_layer,
                    local_plan.page_indices_d(),
                    local_plan.page_indptr_d(),
                    local_origins,
                    global_plan.positions_d(),
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
        }

        let window_left = i32::try_from(self.sliding_window - 1).expect("window fits i32");
        ops::batch_prefill_paged_window_hd256_into(
            ctx,
            &scratch.q_prep,
            self.local_pool.buffer(),
            &self.local_pool.layout().kernel_layout(),
            family_layer,
            local_plan,
            &mut scratch.attn,
            geom.num_q_heads,
            1.0,
            window_left,
        )?;
        attention_epilogue_into(ctx, layer, geom, x, &scratch.attn, epilogue, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn global_layer_serve(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        layer: &Gemma4Layer,
        family_layer: usize,
        seq_len: usize,
        prep: PrepRef<'_>,
        global_plan: &PrefillPagedPlan,
        src: usize,
    ) -> Result<()> {
        let geom = &self.global_geom;
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        anyhow::ensure!(
            layer.attention.v_proj.is_none(),
            "global layer must not carry a v_proj; V is the k_proj fork"
        );
        let TowerScratch {
            attn: scratch,
            epilogue,
            hidden,
        } = tower;
        scratch.set(geom, seq_len);
        let (x, out) = hidden_pair(hidden, src);

        ops::rms_norm_batch_into(
            ctx,
            x,
            &layer.input_layernorm,
            geom.rms_norm_eps,
            &mut scratch.normed_x,
        );
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.q_proj,
            0,
            q_dim,
            &scratch.normed_x,
            &mut scratch.q_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.k_proj,
            0,
            kv_dim,
            &scratch.normed_x,
            &mut scratch.k_states,
        )?;

        // The prep writes both K and the weightless-normed V fork from the
        // one raw K read — no D2D fork copy on the serving path.
        match prep {
            PrepRef::Single { start_pos, .. } => {
                ops::qk_norm_partial_rope_paged_prefill_hd512_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &mut scratch.q_prep,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    global_plan.page_indices_d(),
                    start_pos,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
            PrepRef::Batched { .. } => {
                ops::qk_norm_partial_rope_paged_decode_hd512_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &mut scratch.q_prep,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    global_plan.page_indices_d(),
                    global_plan.page_indptr_d(),
                    global_plan.positions_d(),
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
        }

        ops::batch_prefill_paged_hd512_into(
            ctx,
            &scratch.q_prep,
            self.global_pool.buffer(),
            &self.global_pool.layout().kernel_layout(),
            family_layer,
            global_plan,
            &mut scratch.attn,
            geom.num_q_heads,
            1.0,
        )?;
        attention_epilogue_into(ctx, layer, geom, x, &scratch.attn, epilogue, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_decode_batch(
        &self,
        ctx: &DeviceContext,
        local_plan: &mut PrefillPagedPlan,
        global_plan: &mut PrefillPagedPlan,
        origins_slot: &mut CudaSlice<i32>,
        kvs: &[&mut GemmaKv],
    ) -> Result<()> {
        let batch = kvs.len();
        let page = self.local_pool.layout().page_size;
        let mut local_rows: Vec<Vec<i32>> = Vec::with_capacity(batch);
        let mut local_last = Vec::with_capacity(batch);
        let mut local_start = Vec::with_capacity(batch);
        let mut global_rows: Vec<Vec<i32>> = Vec::with_capacity(batch);
        let mut global_last = Vec::with_capacity(batch);
        let mut global_start = Vec::with_capacity(batch);
        let mut local_origins = Vec::with_capacity(batch);
        let ones = vec![1usize; batch];

        for kv in kvs {
            let start_pos = kv.local.seq_len();
            let kv_len = start_pos + 1;
            self.check_step_bounds(kv, kv_len)?;
            // The local plan lives in cache-relative coordinates, the same
            // way a single-request step builds it.
            let origin_tokens = kv.local.origin_pages() * page;
            let rel_kv_len = kv_len
                .checked_sub(origin_tokens)
                .context("the resident window starts past the step's frontier")?;
            let rel_start = start_pos
                .checked_sub(origin_tokens)
                .context("the step starts before the resident window")?;
            let row = kv.local.page_row();
            anyhow::ensure!(
                row.len() == rel_kv_len.div_ceil(page),
                "local resident row of {} pages against {rel_kv_len} tokens",
                row.len()
            );
            let rel_last_page = if rel_kv_len.is_multiple_of(page) {
                page
            } else {
                rel_kv_len % page
            };
            // desc_for_len checks the held pages against this step's length,
            // so its last-page rule and the state's page ids describe the
            // same rows.
            let global_desc = kv.global.desc_for_len(kv_len)?;
            local_origins.push(i32::try_from(kv.local.origin_pages()).context("origin fits i32")?);
            local_last.push(rel_last_page);
            local_start.push(rel_start);
            local_rows.push(row);
            global_last.push(global_desc.last_page_len());
            // The global family never releases its front, so it plans in
            // absolute coordinates and its per-token positions are the step's
            // — which is what both preps read.
            global_start.push(start_pos);
            global_rows.push(kv.global.page_indices_i32());
        }

        local_plan.update_batch_with_cta_tile_q(
            ctx,
            &local_rows,
            &local_last,
            &local_start,
            &ones,
            self.local_geom.num_q_heads,
            self.local_geom.num_kv_heads,
            self.local_geom.head_dim,
            0,
        )?;
        global_plan.update_batch_with_cta_tile_q(
            ctx,
            &global_rows,
            &global_last,
            &global_start,
            &ones,
            self.global_geom.num_q_heads,
            self.global_geom.num_kv_heads,
            self.global_geom.head_dim,
            0,
        )?;
        upload_prefix(ctx, origins_slot, &local_origins)
    }

    /// The stream that built the pools is the one that ordered every page
    /// write already in them, so a step on any other stream has no ordering
    /// against the KV it is about to read.
    fn check_stream(&self, ctx: &DeviceContext) -> Result<()> {
        anyhow::ensure!(
            std::sync::Arc::ptr_eq(&ctx.stream, self.local_pool.buffer().stream()),
            "a step must use the DeviceContext stream that constructed this GemmaServe"
        );
        Ok(())
    }

    /// Returns the hidden slot holding the tower's output.
    #[allow(clippy::too_many_arguments)]
    fn run_tower(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        ids: &CudaSlice<u32>,
        seq_len: usize,
        local_plan: &PrefillPagedPlan,
        global_plan: &PrefillPagedPlan,
        prep: PrepRef<'_>,
    ) -> Result<usize> {
        let weights = &self.weights;
        ops::embedding_batch(ctx, &weights.embed_tokens, ids, &mut tower.hidden[0])?;
        ops::scale_bf16_in_place(
            ctx,
            &mut tower.hidden[0],
            embed_scale_bf16(self.local_geom.hidden_size),
        )?;
        let mut src = 0usize;
        for (index, kind) in weights.config.layer_types.iter().enumerate() {
            let layer = &weights.layers[index];
            let family_layer = self.family_index[index];
            match kind {
                LayerKind::Sliding => {
                    self.local_layer_serve(
                        ctx,
                        tower,
                        layer,
                        family_layer,
                        seq_len,
                        prep,
                        local_plan,
                        global_plan,
                        src,
                    )?;
                }
                LayerKind::Global => {
                    self.global_layer_serve(
                        ctx,
                        tower,
                        layer,
                        family_layer,
                        seq_len,
                        prep,
                        global_plan,
                        src,
                    )?;
                }
            }
            src ^= 1;
        }
        Ok(src)
    }

    /// One batched decode step: each request advances a single token and the
    /// batch shares every layer's weight pass. Row `r` of the returned logits
    /// is request `r`'s next-token distribution; they live in the arena and
    /// are valid until its next step.
    pub(crate) fn decode_batch_step<'a>(
        &self,
        ctx: &DeviceContext,
        arena: &'a mut StepArena,
        kvs: &mut [&mut GemmaKv],
        tokens: &[u32],
    ) -> Result<&'a mut HiddenStates> {
        let batch = kvs.len();
        anyhow::ensure!(batch > 0, "a decode batch needs at least one request");
        anyhow::ensure!(
            tokens.len() == batch,
            "decode batch has {batch} requests but {} tokens",
            tokens.len()
        );
        self.check_stream(ctx)?;
        let weights = &self.weights;
        validate_tokens(weights, self.local_geom.hidden_size, tokens)?;

        arena.open(ctx, batch)?;
        let StepArena {
            tower,
            local_plan,
            global_plan,
            local_origins,
            ids,
            head_normed,
            logits,
            ..
        } = arena;
        self.plan_decode_batch(ctx, local_plan, global_plan, local_origins, kvs)?;
        upload_prefix(ctx, ids, tokens)?;
        let src = self.run_tower(
            ctx,
            tower,
            ids,
            batch,
            local_plan,
            global_plan,
            PrepRef::Batched { local_origins },
        )?;
        // Every row is its own request's last position, so the whole batch is
        // the head's input.
        logits_tail_into(
            ctx,
            weights,
            &tower.hidden[src],
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
            head_normed,
            logits,
        )?;
        // Append-then-attend, per request: the batch shared a step, but each
        // request owns its own frontier and its own released front.
        for kv in kvs.iter_mut() {
            kv.local.advance_and_release(1, self.sliding_window)?;
            kv.global.advance(1);
        }
        Ok(logits)
    }

    pub(crate) fn step(
        &self,
        ctx: &DeviceContext,
        kv: &mut GemmaKv,
        tokens: &[u32],
        span: LogitsSpan,
    ) -> Result<HiddenStates> {
        let seq_len = tokens.len();
        anyhow::ensure!(seq_len > 0, "step needs at least one token");
        self.check_stream(ctx)?;
        let weights = &self.weights;
        validate_tokens(weights, self.local_geom.hidden_size, tokens)?;
        let start_pos = kv.local.seq_len();
        self.check_step_bounds(kv, start_pos + seq_len)?;
        let plan = self.plan_step(ctx, kv, start_pos, seq_len)?;
        log::debug!(
            "gemma4 step: start_pos {start_pos} seq_len {seq_len} pages local {} global {}",
            kv.local.held_pages(),
            kv.global.held_pages()
        );

        let mut tower = TowerScratch::new(ctx, &self.local_geom, &self.global_geom, seq_len)?;
        let ids = ctx
            .stream
            .clone_htod(tokens)
            .map_err(|e| anyhow::anyhow!("token ids H2D failed: {e}"))?;
        let src = self.run_tower(
            ctx,
            &mut tower,
            &ids,
            seq_len,
            &plan.local_plan,
            &plan.global_plan,
            PrepRef::Single {
                start_pos: plan.start_pos,
                local_page_origin: plan.local_page_origin,
            },
        )?;
        let hidden = &tower.hidden[src];
        // Projecting every prompt row through the 262k LM head materializes
        // half a gigabyte at this path's 1024-token ceiling, so the full span
        // is for callers that actually read every position.
        let mut last_row_slot = None;
        let head_input = match span {
            LogitsSpan::All => hidden,
            LogitsSpan::LastRow => {
                let mut last = HiddenStates::zeros(ctx, hidden.hidden_dim, 1)?;
                ops::copy_hidden_token_range_into(ctx, hidden, seq_len - 1, &mut last, 0, 1)?;
                &*last_row_slot.insert(last)
            }
        };
        let logits = logits_tail(
            ctx,
            weights,
            head_input,
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
        )?;
        // Append-then-attend: the old window and the new tokens were both
        // resident through the layers above, so only now is the front safe to
        // release. The local move is settled first because it is the only
        // fallible one — the global advance cannot fail, so nothing here can
        // leave one family ahead of the other.
        #[cfg(test)]
        if self.release_enabled {
            kv.local.advance_and_release(seq_len, self.sliding_window)?;
        } else {
            kv.local.advance(seq_len);
        }
        #[cfg(not(test))]
        kv.local.advance_and_release(seq_len, self.sliding_window)?;
        kv.global.advance(seq_len);
        Ok(logits)
    }
}

#[path = "serve_oracle.rs"]
#[cfg(test)]
mod oracle;
