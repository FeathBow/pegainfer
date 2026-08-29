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

/// `f32::max` drops a NaN operand, so a non-finite production value would
/// vanish from a plain fold; it counts as infinite error here instead.
fn abs_gap(a: f32, b: f32) -> f32 {
    if a.is_finite() && b.is_finite() {
        (a - b).abs()
    } else {
        f32::INFINITY
    }
}

fn relative_gap(mine: &[f32], reference: &[f32]) -> f32 {
    let scale = reference
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    mine.iter()
        .zip(reference)
        .fold(0.0f32, |acc, (a, b)| acc.max(abs_gap(*a, *b)))
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
        actual.index.starts_with(&expected.index),
        "{label}: router picks moved"
    );
    assert!(
        actual.weight.starts_with(&expected.weight),
        "{label}: router weights moved"
    );
    assert!(
        actual.gate.starts_with(&expected.gate),
        "{label}: gate bytes moved"
    );
    assert!(
        actual.block.starts_with(&expected.block),
        "{label}: block bytes moved"
    );
}

struct RoutedReference {
    experts: usize,
    top_k: usize,
    hidden: usize,
    width: usize,
    /// Router logits per row, rounded to bf16 as the router GEMM stores them.
    logits: Vec<f32>,
    index: Vec<i32>,
    weight: Vec<f32>,
    gate: Vec<f32>,
    block: Vec<f32>,
}

/// One bf16 spacing at `x`'s magnitude. Two logits this close land on the
/// same or adjacent bf16 values under a different accumulation order, so a
/// top-k that swaps them is a tie decided by rounding, not a different
/// computation.
fn bf16_ulp(x: f32) -> f32 {
    let magnitude = x.abs().max(f32::MIN_POSITIVE);
    2f32.powi(magnitude.log2().floor() as i32 - 7)
}

fn assert_matches_reference(label: &str, capture: &RoutedCapture, reference: &RoutedReference) {
    const WEIGHT_TOLERANCE: f32 = 5e-3;
    const RELATIVE_TOLERANCE: f32 = 2e-2;
    // Rows whose picks differ only by a bf16-level tie leave the numeric
    // comparison (their expert set is not the reference's); more than one
    // row in this many says the input, not the arithmetic, is at fault.
    const TIED_ROWS_ONE_IN: usize = 8;

    let (experts, top_k, hidden, width) = (
        reference.experts,
        reference.top_k,
        reference.hidden,
        reference.width,
    );
    let rows = capture.index.len() / top_k;
    let mut tied_rows = Vec::new();
    for row in 0..rows {
        let slots = row * top_k..(row + 1) * top_k;
        let mine = &capture.index[slots.clone()];
        let theirs = &reference.index[slots.clone()];
        if mine != theirs {
            let logits = &reference.logits[row * experts..(row + 1) * experts];
            let only_mine: Vec<i32> = mine
                .iter()
                .copied()
                .filter(|e| !theirs.contains(e))
                .collect();
            let only_theirs: Vec<i32> = theirs
                .iter()
                .copied()
                .filter(|e| !mine.contains(e))
                .collect();
            let tie = !only_mine.is_empty()
                && only_mine.len() == only_theirs.len()
                && only_mine.iter().all(|a| {
                    only_theirs.iter().all(|b| {
                        let (la, lb) = (logits[*a as usize], logits[*b as usize]);
                        (la - lb).abs() <= bf16_ulp(la.abs().max(lb.abs()))
                    })
                });
            assert!(
                tie,
                "{label}: row {row} router picks {mine:?} differ from the reference {theirs:?} \
                 beyond a bf16 tie"
            );
            tied_rows.push(row);
            continue;
        }
        let weight_gap = capture.weight[slots.clone()]
            .iter()
            .zip(&reference.weight[slots.clone()])
            .fold(0.0f32, |acc, (a, b)| acc.max(abs_gap(*a, *b)));
        assert!(
            weight_gap <= WEIGHT_TOLERANCE,
            "{label}: row {row} router weights differ by {weight_gap:.3e}"
        );
        let gate_span = row * top_k * width..(row + 1) * top_k * width;
        let gate = capture.gate[gate_span.clone()]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>();
        let gate_gap = relative_gap(&gate, &reference.gate[gate_span]);
        assert!(
            gate_gap <= RELATIVE_TOLERANCE,
            "{label}: row {row} expert GEMM differs from the widened reference by {gate_gap:.3e}"
        );
        let block_span = row * hidden..(row + 1) * hidden;
        let block = capture.block[block_span.clone()]
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>();
        let block_gap = relative_gap(&block, &reference.block[block_span]);
        assert!(
            block_gap <= RELATIVE_TOLERANCE,
            "{label}: row {row} combined block differs from the reference by {block_gap:.3e}"
        );
    }
    assert!(
        tied_rows.len() <= rows / TIED_ROWS_ONE_IN,
        "{label}: {} of {rows} rows tied at bf16: {tied_rows:?}",
        tied_rows.len()
    );
    if !tied_rows.is_empty() {
        eprintln!("{label}: rows {tied_rows:?} tied at bf16 and left the numeric comparison");
    }
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

    const NARROW_TARGET_ROWS: usize = 4;
    const COMPANION_ROWS: usize = NARROW_TARGET_ROWS + 1;
    const ROOMY_SCRATCH_ROWS: usize = 8;
    const COARSE_TARGET_ROWS: usize = 40;

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
    let capture = |residual_host: &[bf16], dense_host: &[bf16], scratch: &mut MoeScratch| {
        let active_rows = residual_host.len() / hidden;
        let active_slots = active_rows * top_k;
        let residual =
            HiddenStates::from_host(&ctx, residual_host, hidden, active_rows).expect("residual");
        let dense = HiddenStates::from_host(&ctx, dense_host, hidden, active_rows).expect("dense");
        let mut out = HiddenStates::zeros(&ctx, hidden, active_rows).expect("out");
        moe_into(&ctx, moe, &geom, &residual, &dense, scratch, &mut out).expect("routed block");
        RoutedCapture {
            index: ctx
                .stream
                .clone_dtoh(&scratch.index.slice(..active_slots))
                .expect("index"),
            weight: ctx
                .stream
                .clone_dtoh(&scratch.weight.slice(..active_slots))
                .expect("weight"),
            gate: ctx
                .stream
                .clone_dtoh(&scratch.routed_gate.data.slice(..active_slots * width))
                .expect("routed gate"),
            block: ctx
                .stream
                .clone_dtoh(&out.data.slice(..active_rows * hidden))
                .expect("out"),
        }
    };
    let run = |residual_host: &[bf16], dense_host: &[bf16], scratch_rows: usize| {
        let mut scratch = MoeScratch::new(&ctx, &geom, scratch_rows).expect("scratch");
        capture(residual_host, dense_host, &mut scratch)
    };

    let residual_host = sample(0, NARROW_TARGET_ROWS);
    let dense_host = sample(1, NARROW_TARGET_ROWS);
    let baseline = run(&residual_host, &dense_host, NARROW_TARGET_ROWS);
    let roomy = run(&residual_host, &dense_host, ROOMY_SCRATCH_ROWS);
    let companion = run(
        &sample(0, COMPANION_ROWS),
        &sample(1, COMPANION_ROWS),
        ROOMY_SCRATCH_ROWS,
    );
    assert_same_target("scratch capacity", &baseline, &roomy);
    assert_same_target("companion route", &baseline, &companion);

    let coarse_residual_host = sample(0, COARSE_TARGET_ROWS);
    let coarse_dense_host = sample(1, COARSE_TARGET_ROWS);
    let mut coarse_scratch =
        MoeScratch::new(&ctx, &geom, COARSE_TARGET_ROWS).expect("coarse scratch");
    let coarse = capture(
        &coarse_residual_host,
        &coarse_dense_host,
        &mut coarse_scratch,
    );
    let reused = capture(&residual_host, &dense_host, &mut coarse_scratch);
    assert_same_target("block 16 after block 64", &baseline, &reused);

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

    let residual_f32: Vec<f32> = coarse_residual_host.iter().map(|x| x.to_f32()).collect();
    let dense_f32: Vec<f32> = coarse_dense_host.iter().map(|x| x.to_f32()).collect();

    let reference_slots = COARSE_TARGET_ROWS * top_k;
    let mut reference_index = vec![0i32; reference_slots];
    let mut reference_weight = vec![0.0f32; reference_slots];
    let mut reference_gate = vec![0.0f32; reference_slots * width];
    let mut reference_block = vec![0.0f32; COARSE_TARGET_ROWS * hidden];
    let mut reference_logits = vec![0.0f32; COARSE_TARGET_ROWS * experts_out];
    for row in 0..COARSE_TARGET_ROWS {
        let residual_row = &residual_f32[row * hidden..(row + 1) * hidden];
        // The device rounds the router's input to bf16 twice — the norm's
        // store and the standalone scalar multiply — and stores its logits as
        // bf16; the reference rounds at the same three points so a top-k
        // over near-equal logits is decided on the same values.
        let scale = (hidden as f32).sqrt().recip();
        let router_in: Vec<f32> = rms(residual_row, Some(&router_scale), eps)
            .iter()
            .map(|v| bf16::from_f32(bf16::from_f32(*v).to_f32() * scale).to_f32())
            .collect();
        let logits: Vec<f32> = (0..experts_out)
            .map(|expert| {
                let logit: f32 = (0..hidden)
                    .map(|i| router_in[i] * router_proj[expert * hidden + i])
                    .sum();
                bf16::from_f32(logit).to_f32()
            })
            .collect();
        reference_logits[row * experts_out..(row + 1) * experts_out].copy_from_slice(&logits);
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

    // The three numeric bounds sit an order of magnitude above what bf16
    // accumulation costs at these widths, so a breach is a different
    // computation rather than a different rounding.
    let reference = RoutedReference {
        experts: experts_out,
        top_k,
        hidden,
        width,
        logits: reference_logits,
        index: reference_index,
        weight: reference_weight,
        gate: reference_gate,
        block: reference_block,
    };
    assert_matches_reference("4-row target", &baseline, &reference);
    assert_matches_reference("40-row target", &coarse, &reference);
}
