//! KV-backed serving forward: two KV families, prefill and decode, and
//! atomic dual-pool admission.
//!
//! The layer forwards take a `GemmaStepPlan` and the pools this module owns,
//! never a request's state. Both coordinate systems (absolute positions for
//! RoPE, cache-relative slots for the paged scatter) coincide below the
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
use crate::forward::validate_tokens;
use crate::kv::GemmaKv;
use crate::kv::PAGE_SIZE;
use crate::kv::SlidingLocalKv;
use crate::layer::LayerGeometry;
use crate::layer::attention_epilogue;
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
enum StepPrep {
    Single {
        start_pos: usize,
        /// Absolute page the local resident row starts at. The preps shift
        /// only the row index by it; RoPE stays on absolute positions.
        local_page_origin: usize,
    },
    /// One row per request. Each row's page window and absolute position are
    /// the ones its family's plan already carries; what the plans do not
    /// carry is the sliding family's released front, one page index per row.
    Batched { local_origins: CudaSlice<i32> },
}

pub(crate) struct GemmaStepPlan {
    pub(crate) seq_len: usize,
    local_plan: PrefillPagedPlan,
    global_plan: PrefillPagedPlan,
    prep: StepPrep,
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
            seq_len,
            local_plan,
            global_plan,
            prep: StepPrep::Single {
                start_pos,
                local_page_origin: kv.local.origin_pages(),
            },
        })
    }

    fn local_layer_serve(
        &self,
        ctx: &DeviceContext,
        layer: &Gemma4Layer,
        family_layer: usize,
        x: &HiddenStates,
        plan: &GemmaStepPlan,
    ) -> Result<HiddenStates> {
        let geom = &self.local_geom;
        let seq_len = plan.seq_len;
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        let v_proj = layer
            .attention
            .v_proj
            .as_ref()
            .context("local layer requires v_proj")?;

        let mut normed_x = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
        ops::rms_norm_batch_into(
            ctx,
            x,
            &layer.input_layernorm,
            geom.rms_norm_eps,
            &mut normed_x,
        );
        let mut q_states = HiddenStates::zeros(ctx, q_dim, seq_len)?;
        let mut k_states = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
        let mut v_states = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.q_proj,
            0,
            q_dim,
            &normed_x,
            &mut q_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.k_proj,
            0,
            kv_dim,
            &normed_x,
            &mut k_states,
        )?;
        ops::gemm_rows_into_checked(ctx, v_proj, 0, kv_dim, &normed_x, &mut v_states)?;

        let mut q_prep = HiddenStates::zeros(ctx, q_dim, seq_len)?;
        match &plan.prep {
            StepPrep::Single {
                start_pos,
                local_page_origin,
            } => {
                ops::qkv_norm_rope_paged_prefill_hd256_plain_into(
                    ctx,
                    &q_states,
                    &k_states,
                    &v_states,
                    &mut q_prep,
                    self.local_pool.buffer(),
                    &self.local_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.sliding_cos,
                    &self.sliding_sin,
                    family_layer,
                    plan.local_plan.page_indices_d(),
                    *local_page_origin,
                    *start_pos,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
            StepPrep::Batched { local_origins } => {
                ops::qkv_norm_rope_paged_decode_hd256_plain_into(
                    ctx,
                    &q_states,
                    &k_states,
                    &v_states,
                    &mut q_prep,
                    self.local_pool.buffer(),
                    &self.local_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.sliding_cos,
                    &self.sliding_sin,
                    family_layer,
                    plan.local_plan.page_indices_d(),
                    plan.local_plan.page_indptr_d(),
                    local_origins,
                    plan.global_plan.positions_d(),
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
        }

        let mut attn = HiddenStates::zeros(ctx, q_dim, seq_len)?;
        let window_left = i32::try_from(self.sliding_window - 1).expect("window fits i32");
        ops::batch_prefill_paged_window_hd256_into(
            ctx,
            &q_prep,
            self.local_pool.buffer(),
            &self.local_pool.layout().kernel_layout(),
            family_layer,
            &plan.local_plan,
            &mut attn,
            geom.num_q_heads,
            1.0,
            window_left,
        )?;
        attention_epilogue(ctx, layer, geom, x, &attn)
    }

    fn global_layer_serve(
        &self,
        ctx: &DeviceContext,
        layer: &Gemma4Layer,
        family_layer: usize,
        x: &HiddenStates,
        plan: &GemmaStepPlan,
    ) -> Result<HiddenStates> {
        let geom = &self.global_geom;
        let seq_len = plan.seq_len;
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        anyhow::ensure!(
            layer.attention.v_proj.is_none(),
            "global layer must not carry a v_proj; V is the k_proj fork"
        );

        let mut normed_x = HiddenStates::zeros(ctx, geom.hidden_size, seq_len)?;
        ops::rms_norm_batch_into(
            ctx,
            x,
            &layer.input_layernorm,
            geom.rms_norm_eps,
            &mut normed_x,
        );
        let mut q_states = HiddenStates::zeros(ctx, q_dim, seq_len)?;
        let mut k_states = HiddenStates::zeros(ctx, kv_dim, seq_len)?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.q_proj,
            0,
            q_dim,
            &normed_x,
            &mut q_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.k_proj,
            0,
            kv_dim,
            &normed_x,
            &mut k_states,
        )?;

        // The prep writes both K and the weightless-normed V fork from the
        // one raw K read — no D2D fork copy on the serving path.
        let mut q_prep = HiddenStates::zeros(ctx, q_dim, seq_len)?;
        match &plan.prep {
            StepPrep::Single { start_pos, .. } => {
                ops::qk_norm_partial_rope_paged_prefill_hd512_into(
                    ctx,
                    &q_states,
                    &k_states,
                    &mut q_prep,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    plan.global_plan.page_indices_d(),
                    *start_pos,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
            StepPrep::Batched { .. } => {
                ops::qk_norm_partial_rope_paged_decode_hd512_into(
                    ctx,
                    &q_states,
                    &k_states,
                    &mut q_prep,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    plan.global_plan.page_indices_d(),
                    plan.global_plan.page_indptr_d(),
                    plan.global_plan.positions_d(),
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
        }

        let mut attn = HiddenStates::zeros(ctx, q_dim, seq_len)?;
        ops::batch_prefill_paged_hd512_into(
            ctx,
            &q_prep,
            self.global_pool.buffer(),
            &self.global_pool.layout().kernel_layout(),
            family_layer,
            &plan.global_plan,
            &mut attn,
            geom.num_q_heads,
            1.0,
        )?;
        attention_epilogue(ctx, layer, geom, x, &attn)
    }

    /// Plan one decode step for a whole batch: every request contributes one
    /// row, so both families' plans are ragged and each row's page window and
    /// position ride the plan its family already needs.
    fn plan_decode_batch(
        &self,
        ctx: &DeviceContext,
        kvs: &[&mut GemmaKv],
    ) -> Result<GemmaStepPlan> {
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

        let local_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
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
        let global_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
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

        let local_origins = ctx
            .stream
            .clone_htod(&local_origins)
            .map_err(|err| anyhow::anyhow!("decode batch origins H2D failed: {err}"))?;
        Ok(GemmaStepPlan {
            seq_len: batch,
            local_plan,
            global_plan,
            prep: StepPrep::Batched { local_origins },
        })
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

    /// Embed `tokens` and run every layer. Single and batched steps differ in
    /// the plan they hand down, not in the tower they walk.
    fn run_tower(
        &self,
        ctx: &DeviceContext,
        tokens: &[u32],
        plan: &GemmaStepPlan,
    ) -> Result<HiddenStates> {
        let weights = &self.weights;
        let ids = ctx
            .stream
            .clone_htod(tokens)
            .map_err(|err| anyhow::anyhow!("token ids H2D failed: {err}"))?;
        let mut hidden = HiddenStates::zeros(ctx, self.local_geom.hidden_size, tokens.len())?;
        ops::embedding_batch(ctx, &weights.embed_tokens, &ids, &mut hidden)?;
        ops::scale_bf16_in_place(
            ctx,
            &mut hidden,
            embed_scale_bf16(self.local_geom.hidden_size),
        )?;
        for (index, kind) in weights.config.layer_types.iter().enumerate() {
            let layer = &weights.layers[index];
            let family_layer = self.family_index[index];
            hidden = match kind {
                LayerKind::Sliding => {
                    self.local_layer_serve(ctx, layer, family_layer, &hidden, plan)?
                }
                LayerKind::Global => {
                    self.global_layer_serve(ctx, layer, family_layer, &hidden, plan)?
                }
            };
        }
        Ok(hidden)
    }

    /// One batched decode step: each request advances a single token and the
    /// batch shares every layer's weight pass. Row `r` of the returned logits
    /// is request `r`'s next-token distribution.
    pub(crate) fn decode_batch_step(
        &self,
        ctx: &DeviceContext,
        kvs: &mut [&mut GemmaKv],
        tokens: &[u32],
    ) -> Result<HiddenStates> {
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

        let plan = self.plan_decode_batch(ctx, kvs)?;
        let hidden = self.run_tower(ctx, tokens, &plan)?;
        // Every row is its own request's last position, so the whole batch is
        // the head's input.
        let logits = logits_tail(
            ctx,
            weights,
            &hidden,
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
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

        let hidden = self.run_tower(ctx, tokens, &plan)?;
        // Projecting every prompt row through the 262k LM head materializes
        // half a gigabyte at this path's 1024-token ceiling, so the full span
        // is for callers that actually read every position.
        let mut last_row_slot = None;
        let head_input = match span {
            LogitsSpan::All => &hidden,
            LogitsSpan::LastRow => {
                let mut last = HiddenStates::zeros(ctx, hidden.hidden_dim, 1)?;
                ops::copy_hidden_token_range_into(ctx, &hidden, seq_len - 1, &mut last, 0, 1)?;
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
