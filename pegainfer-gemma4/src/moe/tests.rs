use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::mmap_shards;

use super::*;
use crate::manifest::schema::Manifest;
use crate::nvfp4::QuantSource;

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

struct RoutedCapture {
    index: Vec<i32>,
    weight: Vec<f32>,
    gate: Vec<half::bf16>,
    block: Vec<half::bf16>,
}

fn assert_same_target(label: &str, expected: &RoutedCapture, actual: &RoutedCapture) {
    assert!(
        expected.index == actual.index,
        "{label}: router picks moved"
    );
    assert!(
        expected.weight == actual.weight,
        "{label}: router weights moved"
    );
    assert!(expected.gate == actual.gate, "{label}: gate bytes moved");
    assert!(expected.block == actual.block, "{label}: block bytes moved");
}

/// The routed block against the formulas the reference implements, with
/// the router, the expert GEMM and the combined block compared apart:
/// a final-output comparison alone cannot say which of the three moved.
///
/// The reference reads and widens the checkpoint bytes on the host, so it
/// does not share packed/repack/Marlin GPU arithmetic with production.
/// `PEGAINFER_NVFP4_MODEL` names the checkpoint.
#[test]
#[ignore = "requires a GPU and the 26B checkpoint"]
fn the_routed_block_matches_the_reference_formulas() {
    use half::bf16;

    const TARGET_ROWS: usize = 4;
    const COMPANION_ROWS: usize = TARGET_ROWS + 1;
    const ROOMY_SCRATCH_ROWS: usize = 8;

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
    let rows = TARGET_ROWS;
    let slots = rows * top_k;

    let (weights, _) =
        crate::weights::Gemma4Weights::from_safetensors(&model, 0, config).expect("weights");
    let ctx = DeviceContext::new_with_device(0).expect("device");
    let layer = &weights.layers[0];
    let moe = layer.moe.as_ref().expect("layer 0 routes");

    // Inputs both sides rebuild from the same rule, at bf16 precision so
    // neither side starts from a value the other cannot hold.
    let sample = |seed: usize, sample_rows: usize| -> Vec<bf16> {
        (0..sample_rows * hidden)
            .map(|i| bf16::from_f32((((i * 37 + seed * 11) % 199) as f32 - 99.0) / 200.0))
            .collect()
    };
    let run = |residual_host: &[bf16], dense_host: &[bf16], scratch_rows: usize| {
        let active_rows = residual_host.len() / hidden;
        let residual =
            HiddenStates::from_host(&ctx, residual_host, hidden, active_rows).expect("residual");
        let dense = HiddenStates::from_host(&ctx, dense_host, hidden, active_rows).expect("dense");
        let mut scratch = MoeScratch::new(&ctx, &geom, scratch_rows).expect("scratch");
        let mut out = HiddenStates::zeros(&ctx, hidden, active_rows).expect("out");
        moe_into(&ctx, moe, &geom, &residual, &dense, &mut scratch, &mut out)
            .expect("routed block");
        RoutedCapture {
            index: ctx
                .stream
                .clone_dtoh(&scratch.index.slice(..slots))
                .expect("index"),
            weight: ctx
                .stream
                .clone_dtoh(&scratch.weight.slice(..slots))
                .expect("weight"),
            gate: ctx
                .stream
                .clone_dtoh(&scratch.routed_gate.data.slice(..slots * width))
                .expect("routed gate"),
            block: ctx
                .stream
                .clone_dtoh(&out.data.slice(..rows * hidden))
                .expect("out"),
        }
    };

    let residual_host = sample(0, rows);
    let dense_host = sample(1, rows);
    let baseline = run(&residual_host, &dense_host, rows);
    let roomy = run(&residual_host, &dense_host, ROOMY_SCRATCH_ROWS);
    let companion = run(
        &sample(0, COMPANION_ROWS),
        &sample(1, COMPANION_ROWS),
        ROOMY_SCRATCH_ROWS,
    );
    assert_same_target("scratch capacity", &baseline, &roomy);
    assert_same_target("companion route", &baseline, &companion);

    let index = baseline.index;
    let weight = baseline.weight;
    let first_gemm = baseline.gate;
    let produced = baseline.block;

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
