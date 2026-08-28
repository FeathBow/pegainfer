//! The routed-expert half of a Gemma 4 MoE layer.
//!
//! The experts stay packed as NVFP4 on the card and the GEMM reads them that
//! way: widening a layer's experts into bf16 would cost four times the
//! checkpoint in traffic every step, which is what the Marlin kernel exists to
//! avoid. What the loader rewrites once — the weight order, the block-scale
//! encoding — is described in `weights::StackedProjection`.
//!
//! Routing is planned on the device. Grouping the picks into the kernel's
//! fixed-width blocks is a scan the host could do, but it would have to read
//! the picks back first, and a stream synchronize per layer costs more than
//! the scan is worth.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::MarlinDispatch;
use pegainfer_kernels::ops::MoeAlignScratch;
use pegainfer_kernels::ops::gemma4_marlin_nvfp4_moe;
use pegainfer_kernels::ops::gemma4_moe_router_topk_into;
use pegainfer_kernels::ops::gemma4_moe_sum_topk_into;
use pegainfer_kernels::ops::marlin_moe_align_block_size;

use crate::layer::LayerGeometry;
use crate::weights::Gemma4Moe;

/// The kernel's row block. Every expert's picks are padded up to it, so a
/// narrower block wastes less on a thin step and a wider one launches fewer
/// blocks; sixteen is the narrowest the kernel compiles a full table for.
const BLOCK: usize = 16;

/// Marlin's lock array is one int per 64-wide output tile per row block.
const TILE_N: usize = 64;

/// Buffers one [`moe_into`] call needs, sized for the widest step the server
/// admits.
pub(crate) struct MoeScratch {
    max_rows: usize,
    router_in: HiddenStates,
    logits: HiddenStates,
    index: CudaSlice<i32>,
    weight: CudaSlice<f32>,
    sorted_token_ids: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    padded_total: CudaSlice<i32>,
    locks: CudaSlice<i32>,
    c_tmp: CudaSlice<f32>,
    moe_in: HiddenStates,
    routed_gate: HiddenStates,
    routed_up: HiddenStates,
    routed_act: HiddenStates,
    routed_down: HiddenStates,
    expert_out: HiddenStates,
    dense_normed: HiddenStates,
    expert_normed: HiddenStates,
    expert_offsets: CudaSlice<u32>,
    expert_cursor: CudaSlice<u32>,
}

impl MoeScratch {
    pub(crate) fn new(ctx: &DeviceContext, geom: &LayerGeometry, max_rows: usize) -> Result<Self> {
        let moe = geom
            .moe
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: no MoE scratch without a routed config"))?;
        let hidden = |rows| HiddenStates::zeros(ctx, geom.hidden_size, rows);
        let narrow = |rows| HiddenStates::zeros(ctx, moe.intermediate_size, rows);
        let slots = max_rows * moe.top_k;
        // Every expert can leave one block part filled, so the padded total
        // runs past the slot count by that much in the worst case.
        let max_blocks = slots.div_ceil(BLOCK) + moe.num_experts;
        let max_padded = max_blocks * BLOCK;
        Ok(Self {
            max_rows,
            router_in: hidden(max_rows)?,
            logits: HiddenStates::zeros(ctx, moe.num_experts, max_rows)?,
            index: ctx.stream.alloc_zeros::<i32>(slots)?,
            weight: ctx.stream.alloc_zeros::<f32>(slots)?,
            sorted_token_ids: ctx.stream.alloc_zeros::<i32>(max_padded)?,
            expert_ids: ctx.stream.alloc_zeros::<i32>(max_blocks)?,
            padded_total: ctx.stream.alloc_zeros::<i32>(1)?,
            locks: ctx
                .stream
                .alloc_zeros::<i32>(geom.hidden_size / TILE_N * max_blocks)?,
            c_tmp: ctx
                .stream
                .alloc_zeros::<f32>(geom.hidden_size * max_padded)?,
            moe_in: hidden(max_rows)?,
            routed_gate: narrow(slots)?,
            routed_up: narrow(slots)?,
            routed_act: narrow(slots)?,
            routed_down: hidden(slots)?,
            expert_out: hidden(max_rows)?,
            dense_normed: hidden(max_rows)?,
            expert_normed: hidden(max_rows)?,
            expert_offsets: ctx.stream.alloc_zeros::<u32>(moe.num_experts + 1)?,
            expert_cursor: ctx.stream.alloc_zeros::<u32>(moe.num_experts)?,
        })
    }

    fn set_rows(&mut self, rows: usize, top_k: usize) -> Result<()> {
        ensure!(
            rows <= self.max_rows,
            "MoE scratch holds {} rows, not {rows}",
            self.max_rows
        );
        for buf in [
            &mut self.router_in,
            &mut self.logits,
            &mut self.moe_in,
            &mut self.expert_out,
            &mut self.dense_normed,
            &mut self.expert_normed,
        ] {
            buf.seq_len = rows;
        }
        for buf in [
            &mut self.routed_gate,
            &mut self.routed_up,
            &mut self.routed_act,
            &mut self.routed_down,
        ] {
            buf.seq_len = rows * top_k;
        }
        Ok(())
    }
}

/// The routed branch of the feed forward block: `out` receives the sum of the
/// dense output and the expert output, each through its own norm, which the
/// caller still has to put through the shared post-feedforward norm.
///
/// `residual` is the block input — both the router and the experts read it,
/// not the dense branch's output.
pub(crate) fn moe_into(
    ctx: &DeviceContext,
    moe: &Gemma4Moe,
    geom: &LayerGeometry,
    residual: &HiddenStates,
    dense: &HiddenStates,
    scratch: &mut MoeScratch,
    out: &mut HiddenStates,
) -> Result<()> {
    let config = geom
        .moe
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Gemma 4: a routed layer needs a routed config"))?;
    let rows = residual.seq_len;
    scratch.set_rows(rows, config.top_k)?;
    // The GEMM reads the projection's extent from the weights that were
    // loaded, not from the config that described them, so a checkpoint whose
    // experts disagree with its own config stops here rather than reading
    // past a buffer.
    let width = moe.gate.rows;
    let depth = moe.gate.values;
    ensure!(
        width == config.intermediate_size
            && depth == geom.hidden_size
            && moe.up.rows == width
            && moe.up.values == depth
            && moe.down.rows == depth
            && moe.down.values == width,
        "Gemma 4: the routed experts are {width} x {depth}, but the config says {} x {}",
        config.intermediate_size,
        geom.hidden_size
    );

    // The router's own norm carries no scale of its own, so the parameter
    // multiply and the `hidden ** -0.5` that follow it are separate steps.
    ops::rms_norm_batch_into(
        ctx,
        residual,
        &moe.router_scale,
        geom.rms_norm_eps,
        &mut scratch.router_in,
    );
    ops::scale_bf16_in_place(
        ctx,
        &mut scratch.router_in,
        (geom.hidden_size as f32).powf(-0.5),
    )?;
    ops::gemm_rows_into_checked(
        ctx,
        &moe.router_proj,
        0,
        config.num_experts,
        &scratch.router_in,
        &mut scratch.logits,
    )?;
    gemma4_moe_router_topk_into(
        ctx,
        &scratch.logits,
        &moe.router_per_expert_scale,
        config.top_k,
        &mut scratch.index,
        &mut scratch.weight,
    )?;

    let slots = rows * config.top_k;
    marlin_moe_align_block_size(
        ctx,
        &scratch.index,
        rows,
        config.top_k,
        config.num_experts,
        BLOCK,
        &mut MoeAlignScratch {
            sorted_token_ids: &mut scratch.sorted_token_ids,
            expert_ids: &mut scratch.expert_ids,
            num_tokens_post_padded: &mut scratch.padded_total,
            expert_offsets: &mut scratch.expert_offsets,
            expert_cursor: &mut scratch.expert_cursor,
        },
    )?;

    ops::rms_norm_batch_into(
        ctx,
        residual,
        &moe.pre_feedforward_layernorm_2,
        geom.rms_norm_eps,
        &mut scratch.moe_in,
    );

    let gather = MarlinDispatch {
        sorted_token_ids: &scratch.sorted_token_ids,
        expert_ids: &scratch.expert_ids,
        num_tokens_post_padded: &scratch.padded_total,
        topk_weights: &scratch.weight,
        block_size: BLOCK,
        top_k: config.top_k,
        mul_topk_weights: false,
    };
    gemma4_marlin_nvfp4_moe(
        ctx,
        &scratch.moe_in,
        &moe.gate.qweight,
        &moe.gate.scales,
        &moe.gate.global_scales,
        &mut scratch.locks,
        &mut scratch.c_tmp,
        &gather,
        rows,
        width,
        depth,
        &mut scratch.routed_gate,
    )?;
    gemma4_marlin_nvfp4_moe(
        ctx,
        &scratch.moe_in,
        &moe.up.qweight,
        &moe.up.scales,
        &moe.up.global_scales,
        &mut scratch.locks,
        &mut scratch.c_tmp,
        &gather,
        rows,
        width,
        depth,
        &mut scratch.routed_up,
    )?;
    ops::gelu_tanh_mul_batch_into(
        ctx,
        &scratch.routed_gate,
        &scratch.routed_up,
        &mut scratch.routed_act,
    )?;

    // The rows are already one per pick, so the second projection reads them
    // straight through and is the one that applies the router's weights.
    let combine = MarlinDispatch {
        top_k: 1,
        mul_topk_weights: true,
        ..gather
    };
    gemma4_marlin_nvfp4_moe(
        ctx,
        &scratch.routed_act,
        &moe.down.qweight,
        &moe.down.scales,
        &moe.down.global_scales,
        &mut scratch.locks,
        &mut scratch.c_tmp,
        &combine,
        slots,
        depth,
        width,
        &mut scratch.routed_down,
    )?;
    gemma4_moe_sum_topk_into(
        ctx,
        &scratch.routed_down,
        config.top_k,
        &mut scratch.expert_out,
    )?;

    ops::rms_norm_batch_into(
        ctx,
        dense,
        &moe.post_feedforward_layernorm_1,
        geom.rms_norm_eps,
        &mut scratch.dense_normed,
    );
    ops::rms_norm_batch_into(
        ctx,
        &scratch.expert_out,
        &moe.post_feedforward_layernorm_2,
        geom.rms_norm_eps,
        &mut scratch.expert_normed,
    );
    ops::add_batch_into(ctx, &scratch.dense_normed, &scratch.expert_normed, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pegainfer_core::weight_loader::deserialize_shards;
    use pegainfer_core::weight_loader::load_shard_info;
    use pegainfer_core::weight_loader::mmap_shards;

    use super::*;
    use crate::manifest::schema::Manifest;
    use crate::nvfp4::QuantSource;

    /// Row-wise RMS norm with the reference's epsilon, optionally weighted.
    fn rms(row: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inverse = (mean + eps).sqrt().recip();
        row.iter()
            .enumerate()
            .map(|(i, v)| v * inverse * weight.map_or(1.0, |w| w[i]))
            .collect()
    }

    fn gelu_tanh(x: f32) -> f32 {
        let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    /// The largest gap between two spans, relative to the reference's own
    /// magnitude, so a tolerance means the same thing at every scale.
    fn relative_gap(mine: &[f32], reference: &[f32]) -> f32 {
        let scale = reference
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()))
            .max(1e-6);
        mine.iter()
            .zip(reference)
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()))
            / scale
    }

    /// The routed block against the formulas the reference implements, with
    /// the router, the expert GEMM and the combined block compared apart:
    /// a final-output comparison alone cannot say which of the three moved.
    ///
    /// The reference reads the checkpoint's own bytes and widens them on the
    /// host, so it shares no arithmetic with the packed path under test.
    /// `PEGAINFER_NVFP4_MODEL` names the checkpoint.
    #[test]
    #[ignore = "requires a GPU and the 26B checkpoint"]
    fn the_routed_block_matches_the_reference_formulas() {
        use half::bf16;

        let model = std::env::var("PEGAINFER_NVFP4_MODEL")
            .expect("PEGAINFER_NVFP4_MODEL must name the checkpoint directory");

        let config = crate::config::Gemma4Config::from_file(&model).expect("config");
        let eps = config.rms_norm_eps;
        let manifest = Manifest::from_config(&config).expect("manifest");
        let geom = LayerGeometry::local_of(&config);
        let routed = geom.moe.expect("the checkpoint routes");
        let hidden = geom.hidden_size;
        let width = routed.intermediate_size;
        let top_k = routed.top_k;
        let rows = 4;
        let slots = rows * top_k;

        let (weights, _) =
            crate::weights::Gemma4Weights::from_safetensors(&model, 0, config).expect("weights");
        let ctx = DeviceContext::new_with_device(0).expect("device");
        let layer = &weights.layers[0];
        let moe = layer.moe.as_ref().expect("layer 0 routes");

        // Inputs both sides rebuild from the same rule, at bf16 precision so
        // neither side starts from a value the other cannot hold.
        let sample = |seed: usize| -> Vec<bf16> {
            (0..rows * hidden)
                .map(|i| bf16::from_f32((((i * 37 + seed * 11) % 199) as f32 - 99.0) / 200.0))
                .collect()
        };
        let residual_host = sample(0);
        let dense_host = sample(1);
        let mut residual = HiddenStates::zeros(&ctx, hidden, rows).expect("residual");
        let mut dense = HiddenStates::zeros(&ctx, hidden, rows).expect("dense");
        ctx.stream
            .memcpy_htod(&residual_host, &mut residual.data)
            .expect("upload residual");
        ctx.stream
            .memcpy_htod(&dense_host, &mut dense.data)
            .expect("upload dense");

        let mut scratch = MoeScratch::new(&ctx, &geom, rows).expect("scratch");
        let mut out = HiddenStates::zeros(&ctx, hidden, rows).expect("out");
        moe_into(&ctx, moe, &geom, &residual, &dense, &mut scratch, &mut out)
            .expect("routed block");

        let index = ctx
            .stream
            .clone_dtoh(&scratch.index.slice(0..slots))
            .expect("index");
        let weight = ctx
            .stream
            .clone_dtoh(&scratch.weight.slice(0..slots))
            .expect("weight");
        let first_gemm = ctx
            .stream
            .clone_dtoh(&scratch.routed_gate.data)
            .expect("routed gate");
        let produced = ctx.stream.clone_dtoh(&out.data).expect("out");

        let host_vec = |v: &pegainfer_core::tensor::DeviceVec| -> Vec<f32> {
            ctx.stream
                .clone_dtoh(&v.data)
                .expect("norm weight")
                .iter()
                .map(|x: &bf16| x.to_f32())
                .collect()
        };
        let router_scale = host_vec(&moe.router_scale);
        let per_expert_scale = host_vec(&moe.router_per_expert_scale);
        let pre_norm = host_vec(&moe.pre_feedforward_layernorm_2);
        let post_dense_norm = host_vec(&moe.post_feedforward_layernorm_1);
        let post_routed_norm = host_vec(&moe.post_feedforward_layernorm_2);
        let router_proj: Vec<f32> = ctx
            .stream
            .clone_dtoh(&moe.router_proj.data)
            .expect("router proj")
            .iter()
            .map(|x: &bf16| x.to_f32())
            .collect();
        let experts_out = router_proj.len() / hidden;

        // The reference's expert weights come from the checkpoint's bytes,
        // widened on the host rather than read in the packed form.
        let (shard_paths, _) = load_shard_info(&model).expect("shard info");
        let mmaps = mmap_shards(&shard_paths).expect("mmap shards");
        let shards = deserialize_shards(&mmaps).expect("shards");
        let plans = manifest.layers[0]
            .moe
            .as_ref()
            .expect("manifest routes layer 0");
        let widen = |plan: &crate::manifest::schema::QuantMatrix| -> Vec<f32> {
            let (rows, values) = plan.geometry().expect("geometry");
            QuantSource::read(&shards, plan)
                .expect("quant source")
                .widen(rows, values)
                .expect("widen")
        };

        let residual_f32: Vec<f32> = residual_host.iter().map(|x| x.to_f32()).collect();
        let dense_f32: Vec<f32> = dense_host.iter().map(|x| x.to_f32()).collect();

        let mut reference_index = vec![0i32; slots];
        let mut reference_weight = vec![0.0f32; slots];
        let mut reference_gate = vec![0.0f32; slots * width];
        let mut reference_block = vec![0.0f32; rows * hidden];
        for row in 0..rows {
            let residual_row = &residual_f32[row * hidden..(row + 1) * hidden];
            // Router: the scaled norm, the projection, then a softmax whose
            // top picks are renormalised and scaled per expert.
            let router_in = rms(residual_row, Some(&router_scale), eps);
            let scale = (hidden as f32).sqrt().recip();
            let logits: Vec<f32> = (0..experts_out)
                .map(|expert| {
                    (0..hidden)
                        .map(|i| router_in[i] * scale * router_proj[expert * hidden + i])
                        .sum()
                })
                .collect();
            let top = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let exponentials: Vec<f32> = logits.iter().map(|v| (v - top).exp()).collect();
            let total: f32 = exponentials.iter().sum();
            let mut ranked: Vec<usize> = (0..experts_out).collect();
            ranked.sort_by(|a, b| {
                exponentials[*b]
                    .partial_cmp(&exponentials[*a])
                    .expect("finite")
                    .then(a.cmp(b))
            });
            let picked = &ranked[..top_k];
            let picked_total: f32 = picked.iter().map(|e| exponentials[*e] / total).sum();

            let expert_in = rms(residual_row, Some(&pre_norm), eps);
            let mut routed_row = vec![0.0f32; hidden];
            for (pick, &expert) in picked.iter().enumerate() {
                let at = row * top_k + pick;
                reference_index[at] = i32::try_from(expert).expect("expert id");
                let share = (exponentials[expert] / total) / picked_total;
                reference_weight[at] = share * per_expert_scale[expert];

                let gate = widen(&plans.experts[expert].gate);
                let up = widen(&plans.experts[expert].up);
                let down = widen(&plans.experts[expert].down);
                let mut activated = vec![0.0f32; width];
                for column in 0..width {
                    let g: f32 = (0..hidden)
                        .map(|i| expert_in[i] * gate[column * hidden + i])
                        .sum();
                    let u: f32 = (0..hidden)
                        .map(|i| expert_in[i] * up[column * hidden + i])
                        .sum();
                    reference_gate[at * width + column] = g;
                    activated[column] = gelu_tanh(g) * u;
                }
                for (i, slot) in routed_row.iter_mut().enumerate() {
                    let projected: f32 = (0..width)
                        .map(|column| activated[column] * down[i * width + column])
                        .sum();
                    *slot += reference_weight[at] * projected;
                }
            }
            let dense_normed = rms(
                &dense_f32[row * hidden..(row + 1) * hidden],
                Some(&post_dense_norm),
                eps,
            );
            let routed_normed = rms(&routed_row, Some(&post_routed_norm), eps);
            for i in 0..hidden {
                reference_block[row * hidden + i] = dense_normed[i] + routed_normed[i];
            }
        }

        // The picks are a discrete choice and agree exactly or not at all.
        assert_eq!(
            index, reference_index,
            "router picks differ from the reference"
        );
        // The three numeric bounds sit an order of magnitude above what bf16
        // accumulation costs at these widths, so a breach is a different
        // computation rather than a different rounding.
        let weight_gap = weight
            .iter()
            .zip(&reference_weight)
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            weight_gap <= 5e-3,
            "router weights differ by {weight_gap:.3e}"
        );
        let gate_gap = relative_gap(
            &first_gemm[..slots * width]
                .iter()
                .map(|x: &bf16| x.to_f32())
                .collect::<Vec<_>>(),
            &reference_gate,
        );
        assert!(
            gate_gap <= 2e-2,
            "the expert GEMM differs from the widened reference by {gate_gap:.3e}"
        );
        let block_gap = relative_gap(
            &produced[..rows * hidden]
                .iter()
                .map(|x: &bf16| x.to_f32())
                .collect::<Vec<_>>(),
            &reference_block,
        );
        assert!(
            block_gap <= 2e-2,
            "the combined block differs from the reference by {block_gap:.3e}"
        );
    }
}
