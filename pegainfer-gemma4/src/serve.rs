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
//! Batched decode reads the local family through the windowed prefill
//! entry at seq_len 1 and the global family through its native split-KV
//! decode entry. Attention reads are read-only (the prep kernels own the
//! pool writes) with sm_scale 1.0 — Gemma 4 runs unscaled attention.

use anyhow::Context as AnyhowContext;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::Gemma4Config;
use crate::config::LayerKind;
use crate::forward::embed_scale_bf16;
use crate::forward::logits_tail;
use crate::forward::logits_tail_into;
use crate::forward::validate_tokens;
use crate::kv::GemmaKv;
use crate::kv::PAGE_SIZE;
use crate::kv::SlidingLocalKv;
use crate::kv::admit_tokens;
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
        global_plan: &'a PrefillPagedPlan,
    },
    /// One row per request. Each row's local page window rides the local
    /// plan; the global family's window and positions live in the step's
    /// uploaded tables. What neither carries is the sliding family's
    /// released front, one page index per row.
    Batched {
        local_origins: &'a CudaSlice<i32>,
        global_tables: &'a GlobalTables,
    },
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

fn bucket_slot(bucket: usize) -> usize {
    bucket.trailing_zeros() as usize
}

fn hidden_pair(hidden: &mut [HiddenStates; 2], src: usize) -> (&HiddenStates, &mut HiddenStates) {
    let (first, second) = hidden.split_at_mut(1);
    if src == 0 {
        (&first[0], &mut second[0])
    } else {
        (&second[0], &mut first[0])
    }
}

const GLOBAL_SPLIT_CHUNK_TOKENS: usize = 256;

/// The global family's decode tables, uploaded per step: the per-request
/// half feeds the prep, the factor-repeated half feeds the split-KV
/// attention read over the pseudo-requests (see [`global_split_factor`]).
struct GlobalTables {
    pages: CudaSlice<i32>,
    indptr: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    pseudo_pages: CudaSlice<i32>,
    pseudo_indptr: CudaSlice<i32>,
    pseudo_last: CudaSlice<i32>,
}

/// The split-KV plan, refilled per step at graph-stable padded shapes; the
/// chunk size is written to its device slot once at alloc.
struct SplitKvState {
    request_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    chunk_size_d: CudaSlice<i32>,
    o_indptr_d: CudaSlice<i32>,
    valid_mask_d: CudaSlice<u8>,
    tmp_v: CudaSlice<bf16>,
    tmp_s: CudaSlice<f32>,
    /// Chunk-count bound per pseudo-request; a step's padded slot count is
    /// the split factor times its bucket times this.
    cap: usize,
}

pub(crate) struct StepArena {
    tower: TowerScratch,
    local_plan: PrefillPagedPlan,
    global_tables: GlobalTables,
    global_split: SplitKvState,
    local_origins: CudaSlice<i32>,
    ids: CudaSlice<u32>,
    head_normed: HiddenStates,
    logits: HiddenStates,
    /// One graph per power-of-two bucket, at index `log2(bucket)`, captured
    /// by the startup sweep. The captured kernels read and write only
    /// step-stable pointers; per-step change rides the plan and metadata
    /// contents uploaded before launch.
    graphs: Vec<CudaGraphState>,
    graph_enabled: bool,
    /// Floor for the padded bucket. Only the pre-capture sweep raises it, to
    /// drive every bucket from a single dummy request.
    min_bucket: usize,
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

/// How many pseudo-requests the global decode read presents each request
/// as. FlashInfer's decode dispatcher compiles GQA groups {1,2,3,4,8}: a
/// dispatchable group passes through whole, and a non-dispatchable group
/// over one KV head halves into pseudo-requests — an exact memory identity
/// only because MQA gives every query head the same KV head (the 12B
/// global family's 16 over 1). Anything else fails loud.
pub(crate) fn global_split_factor(config: &Gemma4Config) -> Result<usize> {
    const DISPATCHABLE: [usize; 5] = [1, 2, 3, 4, 8];
    let q = config.num_attention_heads;
    let kv = config.num_global_key_value_heads;
    anyhow::ensure!(
        kv > 0 && q.is_multiple_of(kv),
        "global family of {q} query heads over {kv} KV heads is not a whole GQA group"
    );
    let group = q / kv;
    if DISPATCHABLE.contains(&group) {
        return Ok(1);
    }
    if kv == 1 && group.is_multiple_of(2) && DISPATCHABLE.contains(&(group / 2)) {
        return Ok(2);
    }
    anyhow::bail!(
        "the global decode read has no dispatch for {q} query heads over {kv} KV heads \
         (GQA group {group}): supported are groups 1,2,3,4,8 whole, or twice one of \
         those over a single KV head"
    )
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
    global_split_factor: usize,
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
        let global_split_factor = global_split_factor(config)?;
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
            global_split_factor,
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
        graph_enabled: bool,
    ) -> Result<StepArena> {
        anyhow::ensure!(
            max_rows.is_power_of_two(),
            "the arena's {max_rows} rows must be a power of two: steps pad to buckets"
        );
        let group = |geom: &LayerGeometry| geom.num_q_heads / geom.num_kv_heads;
        let alloc = |err: &'static str| {
            move |e: cudarc::driver::DriverError| anyhow::anyhow!("{err} alloc failed: {e}")
        };
        let factor = self.global_split_factor;
        let global_split_cap = self.cos_max_pos.div_ceil(GLOBAL_SPLIT_CHUNK_TOKENS);
        let global_split_slots = factor * max_rows * global_split_cap;
        let global_split_heads = self.global_geom.num_q_heads / factor;
        let mut global_chunk = ctx
            .stream
            .alloc_zeros(1)
            .map_err(alloc("global chunk size"))?;
        ctx.stream
            .memcpy_htod(&[GLOBAL_SPLIT_CHUNK_TOKENS as i32], &mut global_chunk)
            .map_err(|e| anyhow::anyhow!("global chunk-size upload failed: {e}"))?;
        Ok(StepArena {
            tower: TowerScratch::new(ctx, &self.local_geom, &self.global_geom, max_rows)?,
            local_plan: PrefillPagedPlan::new_preallocated(
                ctx,
                max_rows,
                self.local_pool.capacity_pages(),
                max_rows,
                max_rows * group(&self.local_geom),
            )?,
            global_tables: GlobalTables {
                pages: ctx
                    .stream
                    .alloc_zeros(self.global_pool.capacity_pages())
                    .map_err(alloc("global pages"))?,
                indptr: ctx
                    .stream
                    .alloc_zeros(max_rows + 1)
                    .map_err(alloc("global indptr"))?,
                positions: ctx
                    .stream
                    .alloc_zeros(max_rows)
                    .map_err(alloc("global positions"))?,
                pseudo_pages: ctx
                    .stream
                    .alloc_zeros(factor * self.global_pool.capacity_pages())
                    .map_err(alloc("global pseudo pages"))?,
                pseudo_indptr: ctx
                    .stream
                    .alloc_zeros(factor * max_rows + 1)
                    .map_err(alloc("global pseudo indptr"))?,
                pseudo_last: ctx
                    .stream
                    .alloc_zeros(factor * max_rows)
                    .map_err(alloc("global pseudo last-page lens"))?,
            },
            global_split: SplitKvState {
                request_indices_d: ctx
                    .stream
                    .alloc_zeros(global_split_slots)
                    .map_err(alloc("global split request indices"))?,
                kv_tile_indices_d: ctx
                    .stream
                    .alloc_zeros(global_split_slots)
                    .map_err(alloc("global split tile indices"))?,
                chunk_size_d: global_chunk,
                o_indptr_d: ctx
                    .stream
                    .alloc_zeros(factor * max_rows + 1)
                    .map_err(alloc("global split o_indptr"))?,
                valid_mask_d: ctx
                    .stream
                    .alloc_zeros(global_split_slots)
                    .map_err(alloc("global split valid mask"))?,
                tmp_v: ctx
                    .stream
                    .alloc_zeros(
                        global_split_slots * global_split_heads * self.global_geom.head_dim,
                    )
                    .map_err(alloc("global split tmp_v"))?,
                tmp_s: ctx
                    .stream
                    .alloc_zeros(global_split_slots * global_split_heads)
                    .map_err(alloc("global split tmp_s"))?,
                cap: global_split_cap,
            },
            local_origins: ctx.stream.alloc_zeros(max_rows).map_err(alloc("origins"))?,
            ids: ctx.stream.alloc_zeros(max_rows).map_err(alloc("ids"))?,
            head_normed: HiddenStates::zeros(ctx, self.local_geom.hidden_size, max_rows)?,
            logits: HiddenStates::zeros(ctx, self.weights.embed_tokens.rows, max_rows)?,
            graphs: (0..=bucket_slot(max_rows))
                .map(|_| CudaGraphState::new())
                .collect(),
            graph_enabled,
            min_bucket: 1,
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
                ..
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
            PrepRef::Batched {
                local_origins,
                global_tables,
            } => {
                ops::qkv_norm_rope_paged_decode_hd256_plain_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &scratch.v_states,
                    &mut scratch.q_prep,
                    0,
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
                    &global_tables.positions,
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
        split: Option<&mut SplitKvState>,
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
            PrepRef::Single {
                start_pos,
                global_plan,
                ..
            } => {
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
            }
            PrepRef::Batched { global_tables, .. } => {
                let split = split.context("batched global decode needs the split-KV state")?;
                ops::qk_norm_partial_rope_paged_decode_hd512_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &mut scratch.q_prep,
                    0,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    &global_tables.pages,
                    &global_tables.indptr,
                    &global_tables.positions,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
                // A pure reshape: `[rows, q·512]` and `[factor·rows,
                // (q/factor)·512]` are the same memory.
                let factor = self.global_split_factor;
                scratch.q_prep.hidden_dim = q_dim / factor;
                scratch.q_prep.seq_len = factor * seq_len;
                scratch.attn.hidden_dim = q_dim / factor;
                scratch.attn.seq_len = factor * seq_len;
                let meta = ops::Hd512DecodeMetadata::new(
                    &global_tables.pseudo_pages,
                    &global_tables.pseudo_indptr,
                    &global_tables.pseudo_last,
                    &split.request_indices_d,
                    &split.kv_tile_indices_d,
                    &split.chunk_size_d,
                );
                ops::paged_attention_batch_decode_split_kv_hd512_into(
                    ctx,
                    &scratch.q_prep,
                    0,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    family_layer,
                    &meta,
                    &split.o_indptr_d,
                    &split.valid_mask_d,
                    &mut split.tmp_v,
                    &mut split.tmp_s,
                    factor * seq_len * split.cap,
                    &mut scratch.attn,
                    geom.num_q_heads / factor,
                    1.0,
                )?;
                scratch.q_prep.hidden_dim = q_dim;
                scratch.q_prep.seq_len = seq_len;
                scratch.attn.hidden_dim = q_dim;
                scratch.attn.seq_len = seq_len;
            }
        }
        attention_epilogue_into(ctx, layer, geom, x, &scratch.attn, epilogue, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_decode_batch(
        &self,
        ctx: &DeviceContext,
        local_plan: &mut PrefillPagedPlan,
        global_tables: &mut GlobalTables,
        global_split: &mut SplitKvState,
        origins_slot: &mut CudaSlice<i32>,
        kvs: &[&mut GemmaKv],
        padded: usize,
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

        // Pad rows write each pool's reserved padding page — never a real
        // request's KV — at position 0 with a one-token window.
        for _ in batch..padded {
            local_origins.push(0);
            local_rows.push(vec![self.local_pool.padding_page_id()]);
            local_last.push(1);
            local_start.push(0);
            global_rows.push(vec![self.global_pool.padding_page_id()]);
            global_last.push(1);
            global_start.push(0);
        }
        let ones = vec![1usize; padded];
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
        let global_page = self.global_pool.layout().page_size;
        let mut global_pages_cat: Vec<i32> = Vec::new();
        let mut global_indptr = vec![0i32];
        let mut positions = Vec::with_capacity(padded);
        let mut pseudo_pages: Vec<i32> = Vec::new();
        let mut pseudo_indptr = vec![0i32];
        let factor = self.global_split_factor;
        let mut pseudo_last = Vec::with_capacity(factor * padded);
        let mut pseudo_kv_lens = Vec::with_capacity(factor * padded);
        for r in 0..padded {
            let row = &global_rows[r];
            let row_len = i32::try_from(row.len()).context("global pages fit i32")?;
            let last = i32::try_from(global_last[r]).context("global last-page len fits i32")?;
            let kv_len = (row.len() - 1) * global_page + global_last[r];
            positions.push(i32::try_from(global_start[r]).context("position fits i32")?);
            global_indptr.push(global_indptr.last().unwrap() + row_len);
            global_pages_cat.extend_from_slice(row);
            for _ in 0..factor {
                pseudo_pages.extend_from_slice(row);
                pseudo_indptr.push(pseudo_indptr.last().unwrap() + row_len);
                pseudo_last.push(last);
                pseudo_kv_lens.push(kv_len);
            }
        }
        let global_csr = ops::build_split_kv_csr(
            GLOBAL_SPLIT_CHUNK_TOKENS,
            global_split.cap,
            &pseudo_kv_lens,
            factor * padded,
        )?;
        upload_prefix(ctx, &mut global_tables.pages, &global_pages_cat)?;
        upload_prefix(ctx, &mut global_tables.indptr, &global_indptr)?;
        upload_prefix(ctx, &mut global_tables.positions, &positions)?;
        upload_prefix(ctx, &mut global_tables.pseudo_pages, &pseudo_pages)?;
        upload_prefix(ctx, &mut global_tables.pseudo_indptr, &pseudo_indptr)?;
        upload_prefix(ctx, &mut global_tables.pseudo_last, &pseudo_last)?;
        upload_prefix(
            ctx,
            &mut global_split.request_indices_d,
            &global_csr.request_indices,
        )?;
        upload_prefix(
            ctx,
            &mut global_split.kv_tile_indices_d,
            &global_csr.kv_tile_indices,
        )?;
        upload_prefix(ctx, &mut global_split.o_indptr_d, &global_csr.o_indptr)?;
        upload_prefix(
            ctx,
            &mut global_split.valid_mask_d,
            &global_csr.block_valid_mask,
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
        prep: PrepRef<'_>,
        mut global_split: Option<&mut SplitKvState>,
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
                        global_split.as_deref_mut(),
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
        let padded = self.prepare_decode_step(ctx, arena, kvs, tokens)?;
        let StepArena {
            tower,
            local_plan,
            global_tables,
            global_split,
            local_origins,
            ids,
            head_normed,
            logits,
            graphs,
            graph_enabled,
            ..
        } = arena;
        if *graph_enabled {
            let graph = &mut graphs[bucket_slot(padded)];
            anyhow::ensure!(
                graph.is_captured(),
                "no captured graph for bucket {padded}; the pre-capture sweep must cover \
                 every bucket before serving"
            );
            graph.launch_captured(ctx)?;
        } else {
            self.decode_gpu_body(
                ctx,
                tower,
                ids,
                padded,
                local_plan,
                global_tables,
                global_split,
                local_origins,
                head_normed,
                logits,
            )?;
        }
        logits.seq_len = batch;
        // Append-then-attend, per request: the batch shared a step, but each
        // request owns its own frontier and its own released front.
        for kv in kvs.iter_mut() {
            #[cfg(test)]
            if self.release_enabled {
                kv.local.advance_and_release(1, self.sliding_window)?;
            } else {
                kv.local.advance(1);
            }
            #[cfg(not(test))]
            kv.local.advance_and_release(1, self.sliding_window)?;
            kv.global.advance(1);
        }
        Ok(logits)
    }

    /// The decode step's GPU body — embedding through the LM head. A pure
    /// kernel sequence over step-stable pointers: no allocation, no
    /// synchronization, no pool bookkeeping, which is what makes it
    /// capturable.
    #[allow(clippy::too_many_arguments)]
    fn decode_gpu_body(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        ids: &CudaSlice<u32>,
        rows: usize,
        local_plan: &PrefillPagedPlan,
        global_tables: &GlobalTables,
        global_split: &mut SplitKvState,
        local_origins: &CudaSlice<i32>,
        head_normed: &mut HiddenStates,
        logits: &mut HiddenStates,
    ) -> Result<()> {
        let src = self.run_tower(
            ctx,
            tower,
            ids,
            rows,
            local_plan,
            PrepRef::Batched {
                local_origins,
                global_tables,
            },
            Some(global_split),
        )?;
        logits_tail_into(
            ctx,
            &self.weights,
            &tower.hidden[src],
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
            head_normed,
            logits,
        )
    }

    /// One mixed step: a whole admitted prompt (rows `[0..prompt_len)`)
    /// rides the same weight scan as the live decode batch (the row
    /// suffix). Always eager — the prompt length varies per admission, so
    /// this shape never rides a graph; the pure-decode steps around it keep
    /// their bucketed replays. The returned logits hold `batch + 1` rows:
    /// row 0 the prompt's next-token distribution, rows 1.. the decode
    /// batch in order.
    pub(crate) fn mixed_prefill_decode_step<'a>(
        &self,
        ctx: &DeviceContext,
        arena: &'a mut StepArena,
        prefill_kv: &mut GemmaKv,
        prompt: &[u32],
        decode_kvs: &mut [&mut GemmaKv],
        decode_tokens: &[u32],
    ) -> Result<&'a mut HiddenStates> {
        let prefill_len = prompt.len();
        let batch = decode_kvs.len();
        anyhow::ensure!(
            prefill_len > 0 && batch > 0,
            "mixed step needs a prompt ({prefill_len}) and a live decode batch ({batch})"
        );
        anyhow::ensure!(
            decode_tokens.len() == batch,
            "mixed step has {batch} decode requests but {} tokens",
            decode_tokens.len()
        );
        anyhow::ensure!(
            batch < arena.max_rows,
            "mixed step logits rows {} exceed the arena's {} row ceiling",
            batch + 1,
            arena.max_rows
        );
        self.check_stream(ctx)?;
        validate_tokens(&self.weights, self.local_geom.hidden_size, prompt)?;
        validate_tokens(&self.weights, self.local_geom.hidden_size, decode_tokens)?;
        let rows = prefill_len + batch;
        let page = self.local_pool.layout().page_size;

        // The prompt's entry first, then the decode rows — the ragged shape
        // the local plan and the row buffers share. All entries derive from
        // pre-advance state, the same way both pure steps plan.
        let prefill_start = prefill_kv.local.seq_len();
        let prefill_kv_len = prefill_start + prefill_len;
        self.check_step_bounds(prefill_kv, prefill_kv_len)?;
        let origin_tokens = prefill_kv.local.origin_pages() * page;
        let rel_kv_len = prefill_kv_len
            .checked_sub(origin_tokens)
            .context("the resident window starts past the step's frontier")?;
        let rel_start = prefill_start
            .checked_sub(origin_tokens)
            .context("the step starts before the resident window")?;
        let prompt_row = prefill_kv.local.page_row();
        anyhow::ensure!(
            prompt_row.len() == rel_kv_len.div_ceil(page),
            "local resident row of {} pages against {rel_kv_len} tokens",
            prompt_row.len()
        );
        let prompt_rel_last = if rel_kv_len.is_multiple_of(page) {
            page
        } else {
            rel_kv_len % page
        };
        let prefill_origin = prefill_kv.local.origin_pages();

        let mut local_rows = vec![prompt_row];
        let mut local_last = vec![prompt_rel_last];
        let mut local_start = vec![rel_start];
        let mut seq_lens = vec![prefill_len];
        let mut dec_pages_cat: Vec<i32> = Vec::new();
        let mut dec_indptr = vec![0i32];
        let mut positions = Vec::with_capacity(batch);
        let mut local_origins = Vec::with_capacity(batch);
        let mut global_rows: Vec<Vec<i32>> = Vec::with_capacity(batch);
        let mut global_last = Vec::with_capacity(batch);
        for kv in decode_kvs.iter() {
            let start_pos = kv.local.seq_len();
            let kv_len = start_pos + 1;
            self.check_step_bounds(kv, kv_len)?;
            let origin_tokens = kv.local.origin_pages() * page;
            let rel_kv_len = kv_len
                .checked_sub(origin_tokens)
                .context("the resident window starts past the step's frontier")?;
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
            let rel_start = start_pos
                .checked_sub(origin_tokens)
                .context("the step starts before the resident window")?;
            let global_desc = kv.global.desc_for_len(kv_len)?;
            positions.push(i32::try_from(start_pos).context("position fits i32")?);
            local_origins.push(i32::try_from(kv.local.origin_pages()).context("origin fits i32")?);
            dec_indptr.push(
                dec_indptr.last().unwrap()
                    + i32::try_from(row.len()).context("local pages fit i32")?,
            );
            dec_pages_cat.extend_from_slice(&row);
            local_rows.push(row);
            local_last.push(rel_last_page);
            local_start.push(rel_start);
            seq_lens.push(1);
            global_last.push(global_desc.last_page_len());
            global_rows.push(kv.global.page_indices_i32());
        }
        // One ragged local plan covers every row's windowed attention read.
        let local_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            ctx,
            &local_rows,
            &local_last,
            &local_start,
            &seq_lens,
            self.local_geom.num_q_heads,
            self.local_geom.num_kv_heads,
            self.local_geom.head_dim,
            0,
        )?;
        let global_desc = prefill_kv.global.desc_for_len(prefill_kv_len)?;
        let global_prefill_plan = PrefillPagedPlan::new(
            ctx,
            &global_desc,
            prefill_start,
            prefill_len,
            self.global_geom.num_q_heads,
            self.global_geom.num_kv_heads,
            self.global_geom.head_dim,
        )?;
        // Batch-local decode prep tables; transient because the arena's
        // slots stay dedicated to the graphed pure-decode steps.
        let local_pages_t: CudaSlice<i32> = ctx
            .stream
            .clone_htod(&dec_pages_cat)
            .map_err(|e| anyhow::anyhow!("mixed local pages H2D failed: {e}"))?;
        let local_indptr_t: CudaSlice<i32> = ctx
            .stream
            .clone_htod(&dec_indptr)
            .map_err(|e| anyhow::anyhow!("mixed local indptr H2D failed: {e}"))?;

        // The global tables in both shapes, over the decode rows only.
        let global_page = self.global_pool.layout().page_size;
        let mut global_pages_cat: Vec<i32> = Vec::new();
        let mut global_indptr = vec![0i32];
        let factor = self.global_split_factor;
        let mut pseudo_pages: Vec<i32> = Vec::new();
        let mut pseudo_indptr = vec![0i32];
        let mut pseudo_last = Vec::with_capacity(factor * batch);
        let mut pseudo_kv_lens = Vec::with_capacity(factor * batch);
        for r in 0..batch {
            let row = &global_rows[r];
            let row_len = i32::try_from(row.len()).context("global pages fit i32")?;
            let last = i32::try_from(global_last[r]).context("global last-page len fits i32")?;
            let kv_len = (row.len() - 1) * global_page + global_last[r];
            global_indptr.push(global_indptr.last().unwrap() + row_len);
            global_pages_cat.extend_from_slice(row);
            for _ in 0..factor {
                pseudo_pages.extend_from_slice(row);
                pseudo_indptr.push(pseudo_indptr.last().unwrap() + row_len);
                pseudo_last.push(last);
                pseudo_kv_lens.push(kv_len);
            }
        }
        let global_csr = ops::build_split_kv_csr(
            GLOBAL_SPLIT_CHUNK_TOKENS,
            arena.global_split.cap,
            &pseudo_kv_lens,
            factor * batch,
        )?;
        let StepArena {
            global_tables,
            global_split,
            local_origins: origins_slot,
            head_normed,
            logits,
            ..
        } = arena;
        upload_prefix(ctx, &mut global_tables.pages, &global_pages_cat)?;
        upload_prefix(ctx, &mut global_tables.indptr, &global_indptr)?;
        upload_prefix(ctx, &mut global_tables.positions, &positions)?;
        upload_prefix(ctx, &mut global_tables.pseudo_pages, &pseudo_pages)?;
        upload_prefix(ctx, &mut global_tables.pseudo_indptr, &pseudo_indptr)?;
        upload_prefix(ctx, &mut global_tables.pseudo_last, &pseudo_last)?;
        upload_prefix(
            ctx,
            &mut global_split.request_indices_d,
            &global_csr.request_indices,
        )?;
        upload_prefix(
            ctx,
            &mut global_split.kv_tile_indices_d,
            &global_csr.kv_tile_indices,
        )?;
        upload_prefix(ctx, &mut global_split.o_indptr_d, &global_csr.o_indptr)?;
        upload_prefix(
            ctx,
            &mut global_split.valid_mask_d,
            &global_csr.block_valid_mask,
        )?;
        upload_prefix(ctx, origins_slot, &local_origins)?;

        let mut ids_host = Vec::with_capacity(rows);
        ids_host.extend_from_slice(prompt);
        ids_host.extend_from_slice(decode_tokens);
        let ids = ctx
            .stream
            .clone_htod(&ids_host)
            .map_err(|e| anyhow::anyhow!("mixed step ids H2D failed: {e}"))?;
        let mut tower = TowerScratch::new(ctx, &self.local_geom, &self.global_geom, rows)?;
        tower.open(rows)?;
        ops::embedding_batch(ctx, &self.weights.embed_tokens, &ids, &mut tower.hidden[0])?;
        ops::scale_bf16_in_place(
            ctx,
            &mut tower.hidden[0],
            embed_scale_bf16(self.local_geom.hidden_size),
        )?;
        let mut src = 0usize;
        for (index, kind) in self.weights.config.layer_types.iter().enumerate() {
            let layer = &self.weights.layers[index];
            let family_layer = self.family_index[index];
            match kind {
                LayerKind::Sliding => {
                    self.local_layer_mixed(
                        ctx,
                        &mut tower,
                        layer,
                        family_layer,
                        prefill_len,
                        prefill_origin,
                        prefill_start,
                        rows,
                        &local_plan,
                        &local_pages_t,
                        &local_indptr_t,
                        origins_slot,
                        &global_tables.positions,
                        src,
                    )?;
                }
                LayerKind::Global => {
                    self.global_layer_mixed(
                        ctx,
                        &mut tower,
                        layer,
                        family_layer,
                        prefill_len,
                        prefill_start,
                        rows,
                        &global_prefill_plan,
                        global_tables,
                        global_split,
                        src,
                    )?;
                }
            }
            src ^= 1;
        }
        // The sampled rows — the prompt's last plus the decode suffix — are
        // one contiguous range; compact them into the free ping-pong slot
        // and run the batch + 1 rows through the LM head.
        let (x, staging) = hidden_pair(&mut tower.hidden, src);
        staging.seq_len = batch + 1;
        ops::copy_hidden_token_range_into(ctx, x, prefill_len - 1, staging, 0, batch + 1)?;
        logits_tail_into(
            ctx,
            &self.weights,
            staging,
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
            head_normed,
            logits,
        )?;
        logits.seq_len = batch + 1;
        // Append-then-attend, per request, the same way both pure steps
        // settle their frontiers.
        #[cfg(test)]
        if self.release_enabled {
            prefill_kv
                .local
                .advance_and_release(prefill_len, self.sliding_window)?;
        } else {
            prefill_kv.local.advance(prefill_len);
        }
        #[cfg(not(test))]
        prefill_kv
            .local
            .advance_and_release(prefill_len, self.sliding_window)?;
        prefill_kv.global.advance(prefill_len);
        for kv in decode_kvs.iter_mut() {
            #[cfg(test)]
            if self.release_enabled {
                kv.local.advance_and_release(1, self.sliding_window)?;
            } else {
                kv.local.advance(1);
            }
            #[cfg(not(test))]
            kv.local.advance_and_release(1, self.sliding_window)?;
            kv.global.advance(1);
        }
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn local_layer_mixed(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        layer: &Gemma4Layer,
        family_layer: usize,
        prefill_len: usize,
        prefill_origin: usize,
        prefill_start: usize,
        rows: usize,
        local_plan: &PrefillPagedPlan,
        local_pages: &CudaSlice<i32>,
        local_indptr: &CudaSlice<i32>,
        local_origins: &CudaSlice<i32>,
        positions: &CudaSlice<i32>,
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
        scratch.set(geom, rows);
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

        // Prefill segment as prefix views: the ragged plan holds the
        // prompt's pages at the front of the concatenated table, which is
        // the prefill prep's contract.
        scratch.q_states.seq_len = prefill_len;
        scratch.k_states.seq_len = prefill_len;
        scratch.v_states.seq_len = prefill_len;
        scratch.q_prep.seq_len = prefill_len;
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
            prefill_origin,
            prefill_start,
            self.cos_max_pos,
            geom.num_q_heads,
            geom.num_kv_heads,
            geom.head_dim,
            geom.rms_norm_eps,
        )?;
        scratch.q_states.seq_len = rows;
        scratch.k_states.seq_len = rows;
        scratch.v_states.seq_len = rows;
        scratch.q_prep.seq_len = rows;
        ops::qkv_norm_rope_paged_decode_hd256_plain_into(
            ctx,
            &scratch.q_states,
            &scratch.k_states,
            &scratch.v_states,
            &mut scratch.q_prep,
            prefill_len,
            self.local_pool.buffer(),
            &self.local_pool.layout().kernel_layout(),
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.sliding_cos,
            &self.sliding_sin,
            family_layer,
            local_pages,
            local_indptr,
            local_origins,
            positions,
            self.cos_max_pos,
            geom.num_q_heads,
            geom.num_kv_heads,
            geom.head_dim,
            geom.rms_norm_eps,
        )?;

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
    fn global_layer_mixed(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        layer: &Gemma4Layer,
        family_layer: usize,
        prefill_len: usize,
        prefill_start: usize,
        rows: usize,
        global_plan: &PrefillPagedPlan,
        global_tables: &GlobalTables,
        split: &mut SplitKvState,
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
        scratch.set(geom, rows);
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

        scratch.q_states.seq_len = prefill_len;
        scratch.k_states.seq_len = prefill_len;
        scratch.q_prep.seq_len = prefill_len;
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
            prefill_start,
            self.cos_max_pos,
            geom.num_q_heads,
            geom.num_kv_heads,
            geom.head_dim,
            geom.rms_norm_eps,
        )?;
        scratch.q_states.seq_len = rows;
        scratch.k_states.seq_len = rows;
        scratch.q_prep.seq_len = rows;
        ops::qk_norm_partial_rope_paged_decode_hd512_into(
            ctx,
            &scratch.q_states,
            &scratch.k_states,
            &mut scratch.q_prep,
            prefill_len,
            self.global_pool.buffer(),
            &self.global_pool.layout().kernel_layout(),
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.global_cos,
            &self.global_sin,
            family_layer,
            &global_tables.pages,
            &global_tables.indptr,
            &global_tables.positions,
            self.cos_max_pos,
            geom.num_q_heads,
            geom.num_kv_heads,
            geom.head_dim,
            geom.rms_norm_eps,
        )?;

        // The prompt's rows read through the per-prompt plan; the decode
        // suffix reads through the native split entry — the same kernel and
        // chunk plan as a pure decode step.
        scratch.q_prep.seq_len = prefill_len;
        scratch.attn.seq_len = prefill_len;
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
        let batch = rows - prefill_len;
        let factor = self.global_split_factor;
        scratch.q_prep.hidden_dim = q_dim / factor;
        scratch.q_prep.seq_len = factor * rows;
        scratch.attn.hidden_dim = q_dim / factor;
        scratch.attn.seq_len = factor * rows;
        let meta = ops::Hd512DecodeMetadata::new(
            &global_tables.pseudo_pages,
            &global_tables.pseudo_indptr,
            &global_tables.pseudo_last,
            &split.request_indices_d,
            &split.kv_tile_indices_d,
            &split.chunk_size_d,
        );
        ops::paged_attention_batch_decode_split_kv_hd512_into(
            ctx,
            &scratch.q_prep,
            factor * prefill_len,
            self.global_pool.buffer(),
            &self.global_pool.layout().kernel_layout(),
            family_layer,
            &meta,
            &split.o_indptr_d,
            &split.valid_mask_d,
            &mut split.tmp_v,
            &mut split.tmp_s,
            factor * batch * split.cap,
            &mut scratch.attn,
            geom.num_q_heads / factor,
            1.0,
        )?;
        scratch.q_prep.hidden_dim = q_dim;
        scratch.q_prep.seq_len = rows;
        scratch.attn.hidden_dim = q_dim;
        scratch.attn.seq_len = rows;
        attention_epilogue_into(ctx, layer, geom, x, &scratch.attn, epilogue, out)
    }

    /// Host-side preparation a decode step and its capture share: checks,
    /// the bucket, the arena open, the plan refill and the metadata uploads.
    fn prepare_decode_step(
        &self,
        ctx: &DeviceContext,
        arena: &mut StepArena,
        kvs: &[&mut GemmaKv],
        tokens: &[u32],
    ) -> Result<usize> {
        let batch = kvs.len();
        anyhow::ensure!(batch > 0, "a decode batch needs at least one request");
        anyhow::ensure!(
            tokens.len() == batch,
            "decode batch has {batch} requests but {} tokens",
            tokens.len()
        );
        anyhow::ensure!(
            batch <= arena.max_rows,
            "decode batch of {batch} exceeds the arena's {} row ceiling",
            arena.max_rows
        );
        self.check_stream(ctx)?;
        validate_tokens(&self.weights, self.local_geom.hidden_size, tokens)?;

        // A step computes at its power-of-two bucket whether or not graphs
        // are on: padding is part of the numeric contract, which is what
        // keeps a captured replay and the eager escape hatch the same
        // arithmetic at every batch size.
        let padded = batch.next_power_of_two().max(arena.min_bucket);
        arena.open(ctx, padded)?;
        let StepArena {
            local_plan,
            global_tables,
            global_split,
            local_origins,
            ids,
            ..
        } = arena;
        self.plan_decode_batch(
            ctx,
            local_plan,
            global_tables,
            global_split,
            local_origins,
            kvs,
            padded,
        )?;
        let mut padded_tokens = tokens.to_vec();
        padded_tokens.resize(padded, 0);
        upload_prefix(ctx, ids, &padded_tokens)?;
        Ok(padded)
    }

    /// Capture every power-of-two decode graph before serving, then
    /// synchronize, so capture cost and any capture error land here rather
    /// than on the first requests. Each bucket warms one eager pass first —
    /// forcing lazy CUDA and cuBLAS initialization outside capture — records
    /// the body without executing it, then drives the same step through the
    /// serving path, which replays the fresh graph and advances the dummy.
    pub(crate) fn precapture_decode_graphs(
        &self,
        ctx: &DeviceContext,
        arena: &mut StepArena,
    ) -> Result<()> {
        if !arena.graph_enabled {
            return Ok(());
        }
        let mut kv = self.alloc_kv();
        admit_tokens(&self.local_pool, &self.global_pool, &mut kv, 1)?;
        self.step(ctx, &mut kv, &[0], LogitsSpan::LastRow)?;
        let mut bucket = 1usize;
        while bucket <= arena.max_rows {
            arena.min_bucket = bucket;
            admit_tokens(&self.local_pool, &self.global_pool, &mut kv, 1)?;
            {
                let mut kvs: [&mut GemmaKv; 1] = [&mut kv];
                let padded = self.prepare_decode_step(ctx, arena, &kvs, &[0])?;
                let StepArena {
                    tower,
                    local_plan,
                    global_tables,
                    global_split,
                    local_origins,
                    ids,
                    head_normed,
                    logits,
                    graphs,
                    ..
                } = arena;
                self.decode_gpu_body(
                    ctx,
                    tower,
                    ids,
                    padded,
                    local_plan,
                    global_tables,
                    global_split,
                    local_origins,
                    head_normed,
                    logits,
                )?;
                graphs[bucket_slot(padded)].capture_only(ctx, || {
                    self.decode_gpu_body(
                        ctx,
                        tower,
                        ids,
                        padded,
                        local_plan,
                        global_tables,
                        global_split,
                        local_origins,
                        head_normed,
                        logits,
                    )
                })?;
                self.decode_batch_step(ctx, arena, &mut kvs, &[0])?;
            }
            bucket *= 2;
        }
        arena.min_bucket = 1;
        ctx.sync()
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
            PrepRef::Single {
                start_pos: plan.start_pos,
                local_page_origin: plan.local_page_origin,
                global_plan: &plan.global_plan,
            },
            None,
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
