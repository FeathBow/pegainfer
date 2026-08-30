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

/// One dispatch's device results. Logits, picks and weights are small and
/// come back whole; the per-route projections and the block come back for
/// `rows` only, so a thousand-row dispatch does not move tens of MiB to read
/// six rows.
struct RoutedCapture {
    /// The alignment pass's padded row total, which names the block it used.
    padded: usize,
    /// Router logits as the router GEMM stored them, `[rows, experts]`.
    logits: Vec<half::bf16>,
    index: Vec<i32>,
    weight: Vec<f32>,
    rows: Vec<usize>,
    /// Per captured row: the first expert projection of its `top_k` routes.
    gate: std::collections::HashMap<usize, Vec<half::bf16>>,
    /// Per captured row: each route's down projection with the router weight
    /// applied, before the top-k sum and the post norm fold it into the block.
    down: std::collections::HashMap<usize, Vec<half::bf16>>,
    block: std::collections::HashMap<usize, Vec<half::bf16>>,
}

/// The padded total the alignment pass must have produced for these picks at
/// this block: every expert's routes rounded up to the block. It proves which
/// block ran and that the padding matched the routes.
fn assert_padding(label: &str, capture: &RoutedCapture, experts: usize, block: usize) {
    let mut counts = vec![0usize; experts];
    for expert in &capture.index {
        counts[usize::try_from(*expert).expect("expert id")] += 1;
    }
    let expected: usize = counts.iter().map(|c| c.div_ceil(block) * block).sum();
    assert_eq!(
        capture.padded, expected,
        "{label}: padded total is not the {block}-row block's padding of these picks"
    );
}

/// `actual` reproduces `expected` byte for byte on `expected`'s rows.
fn assert_same_target(label: &str, expected: &RoutedCapture, actual: &RoutedCapture) {
    assert!(
        actual.logits.starts_with(&expected.logits),
        "{label}: router logits moved"
    );
    assert!(
        actual.index.starts_with(&expected.index),
        "{label}: router picks moved"
    );
    assert!(
        actual.weight.starts_with(&expected.weight),
        "{label}: router weights moved"
    );
    for row in &expected.rows {
        assert_eq!(
            expected.gate[row], actual.gate[row],
            "{label}: row {row} gate bytes moved"
        );
        assert_eq!(
            expected.down[row], actual.down[row],
            "{label}: row {row} down bytes moved"
        );
        assert_eq!(
            expected.block[row], actual.block[row],
            "{label}: row {row} block bytes moved"
        );
    }
}

/// The block policy's own claim, on its own rows: a route's expert GEMM is
/// the same bytes on the 16-row and the 64-row block. The router projection
/// is a cuBLAS GEMM whose algorithm follows the batch, so a four-row and a
/// thousand-row dispatch may store different logits for the same input row;
/// where a row's pick agrees across the two, the gate bytes must agree.
fn assert_prefix_gate_bytes(
    label: &str,
    narrow: &RoutedCapture,
    coarse: &RoutedCapture,
    top_k: usize,
    width: usize,
) {
    let mut compared = 0usize;
    for row in &narrow.rows {
        for pick in 0..top_k {
            let at = row * top_k + pick;
            if narrow.index[at] != coarse.index[at] {
                continue;
            }
            let span = pick * width..(pick + 1) * width;
            assert_eq!(
                narrow.gate[row][span.clone()],
                coarse.gate[row][span],
                "{label}: row {row} pick {pick} gate bytes differ across the block policy"
            );
            compared += 1;
        }
    }
    assert!(
        compared > 0,
        "{label}: no pick agreed across the two dispatches, nothing was compared"
    );
    eprintln!("{label}: {compared} routes compared byte for byte across the block policy");
}

/// The host side of the block: checkpoint weights widened on the host, the
/// norm vectors, and the expert matrices decoded on demand.
struct HostBlock<'a> {
    hidden: usize,
    width: usize,
    top_k: usize,
    experts: usize,
    eps: f32,
    router_scale: Vec<f32>,
    per_expert_scale: Vec<f32>,
    pre_norm: Vec<f32>,
    post_dense_norm: Vec<f32>,
    post_routed_norm: Vec<f32>,
    router_proj: Vec<f32>,
    plans: &'a crate::manifest::schema::MoeTensors,
    shards: &'a [safetensors::SafeTensors<'a>],
    widened: std::cell::RefCell<std::collections::HashMap<usize, [Vec<f32>; 3]>>,
}

impl HostBlock<'_> {
    fn expert(&self, expert: usize) -> std::cell::Ref<'_, [Vec<f32>; 3]> {
        if !self.widened.borrow().contains_key(&expert) {
            let widen = |plan: &crate::manifest::schema::QuantMatrix| -> Vec<f32> {
                let (rows, values) = plan.geometry().expect("geometry");
                QuantSource::read(self.shards, plan)
                    .expect("quant source")
                    .widen(rows, values)
                    .expect("widen")
            };
            let plan = &self.plans.experts[expert];
            self.widened.borrow_mut().insert(
                expert,
                [widen(&plan.gate), widen(&plan.up), widen(&plan.down)],
            );
        }
        std::cell::Ref::map(self.widened.borrow(), |m| &m[&expert])
    }
}

/// The capture against the block's formulas, one row and one quantity at a
/// time. The host router projection is compared to the device's stored bf16
/// logits with a tolerance; the pick and weight contract is then evaluated
/// exactly on those stored logits, which is what the kernel consumes, so a
/// bf16 logit larger by one spacing is larger, not tied. The expert path is
/// checked for the device's picks: the first projection, each route's
/// weighted down projection before any norm, and the combined block. The
/// relative bounds sit an order of magnitude above bf16 accumulation at these
/// widths, so a breach is a different computation, not a different rounding.
fn assert_matches_reference(
    label: &str,
    capture: &RoutedCapture,
    rows: std::ops::Range<usize>,
    residual_host: &[half::bf16],
    dense_host: &[half::bf16],
    host: &HostBlock<'_>,
) {
    use half::bf16;
    const LOGIT_TOLERANCE: f32 = 1e-2;
    const WEIGHT_TOLERANCE: f32 = 1e-4;
    const RELATIVE_TOLERANCE: f32 = 2e-2;

    let (hidden, width, top_k, experts, eps) =
        (host.hidden, host.width, host.top_k, host.experts, host.eps);
    let residual_f32: Vec<f32> = residual_host.iter().map(|x| x.to_f32()).collect();
    let dense_f32: Vec<f32> = dense_host.iter().map(|x| x.to_f32()).collect();
    for row in rows {
        let residual_row = &residual_f32[row * hidden..(row + 1) * hidden];

        // The host projection rounds where the device does: the norm's
        // store, the scalar multiply, and the stored logits.
        let scale = (hidden as f32).sqrt().recip();
        let router_in: Vec<f32> = rms(residual_row, Some(&host.router_scale), eps)
            .iter()
            .map(|v| bf16::from_f32(bf16::from_f32(*v).to_f32() * scale).to_f32())
            .collect();
        let host_logits: Vec<f32> = (0..experts)
            .map(|expert| {
                let logit: f32 = (0..hidden)
                    .map(|i| router_in[i] * host.router_proj[expert * hidden + i])
                    .sum();
                bf16::from_f32(logit).to_f32()
            })
            .collect();
        let device_logits: Vec<f32> = capture.logits[row * experts..(row + 1) * experts]
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let logit_gap = relative_gap(&device_logits, &host_logits);
        assert!(
            logit_gap <= LOGIT_TOLERANCE,
            "{label}: row {row} router projection differs from the host by {logit_gap:.3e}"
        );

        let top = device_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let exponentials: Vec<f32> = device_logits.iter().map(|v| (v - top).exp()).collect();
        let total: f32 = exponentials.iter().sum();
        let mut ranked: Vec<usize> = (0..experts).collect();
        ranked.sort_by(|a, b| {
            exponentials[*b]
                .partial_cmp(&exponentials[*a])
                .expect("finite")
                .then(a.cmp(b))
        });
        let expected_picks: Vec<i32> = ranked[..top_k]
            .iter()
            .map(|e| i32::try_from(*e).expect("expert id"))
            .collect();
        let picks = &capture.index[row * top_k..(row + 1) * top_k];
        assert_eq!(
            picks, expected_picks,
            "{label}: row {row} router picks differ from the exact contract on the stored logits"
        );
        let picked_total: f32 = picks
            .iter()
            .map(|e| exponentials[*e as usize] / total)
            .sum();
        for (pick, &expert) in picks.iter().enumerate() {
            let expected = (exponentials[expert as usize] / total) / picked_total
                * host.per_expert_scale[expert as usize];
            let gap = abs_gap(capture.weight[row * top_k + pick], expected);
            assert!(
                gap <= WEIGHT_TOLERANCE,
                "{label}: row {row} pick {pick} router weight differs from the contract by {gap:.3e}"
            );
        }

        let expert_in = rms(residual_row, Some(&host.pre_norm), eps);
        let mut routed_row = vec![0.0f32; hidden];
        for (pick, &expert) in picks.iter().enumerate() {
            let at = row * top_k + pick;
            let matrices = host.expert(expert as usize);
            let [gate, up, down] = &*matrices;
            let mut reference_gate = vec![0.0f32; width];
            let mut activated = vec![0.0f32; width];
            for column in 0..width {
                let g: f32 = (0..hidden)
                    .map(|i| expert_in[i] * gate[column * hidden + i])
                    .sum();
                let u: f32 = (0..hidden)
                    .map(|i| expert_in[i] * up[column * hidden + i])
                    .sum();
                reference_gate[column] = g;
                activated[column] = gelu_tanh(g) * u;
            }
            let device_gate: Vec<f32> = capture.gate[&row][pick * width..(pick + 1) * width]
                .iter()
                .map(|x| x.to_f32())
                .collect();
            let gate_gap = relative_gap(&device_gate, &reference_gate);
            assert!(
                gate_gap <= RELATIVE_TOLERANCE,
                "{label}: row {row} pick {pick} expert GEMM differs from the widened reference by \
                 {gate_gap:.3e}"
            );
            // With the device's own weight applied, only the GEMM is compared.
            let weight = capture.weight[at];
            let reference_down: Vec<f32> = (0..hidden)
                .map(|i| {
                    let projected: f32 = (0..width)
                        .map(|column| activated[column] * down[i * width + column])
                        .sum();
                    weight * projected
                })
                .collect();
            let device_down: Vec<f32> = capture.down[&row][pick * hidden..(pick + 1) * hidden]
                .iter()
                .map(|x| x.to_f32())
                .collect();
            let down_gap = relative_gap(&device_down, &reference_down);
            assert!(
                down_gap <= RELATIVE_TOLERANCE,
                "{label}: row {row} pick {pick} down projection differs from the widened reference \
                 by {down_gap:.3e}"
            );
            for (slot, value) in routed_row.iter_mut().zip(&reference_down) {
                *slot += value;
            }
        }
        let dense_normed = rms(
            &dense_f32[row * hidden..(row + 1) * hidden],
            Some(&host.post_dense_norm),
            eps,
        );
        let routed_normed = rms(&routed_row, Some(&host.post_routed_norm), eps);
        let reference_block: Vec<f32> = (0..hidden)
            .map(|i| dense_normed[i] + routed_normed[i])
            .collect();
        let device_block: Vec<f32> = capture.block[&row].iter().map(|x| x.to_f32()).collect();
        let block_gap = relative_gap(&device_block, &reference_block);
        assert!(
            block_gap <= RELATIVE_TOLERANCE,
            "{label}: row {row} combined block differs from the reference by {block_gap:.3e}"
        );
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
    // The production block policy, pinned here rather than derived from it,
    // so a moved constant fails this gate instead of moving its inputs along:
    // the widest dispatch the 16-row block still takes, and the narrowest the
    // 64-row block takes.
    const NARROW_EDGE_ROWS: usize = 1023;
    const COARSE_ROWS: usize = 1024;

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
    assert_eq!(super::marlin_block(NARROW_EDGE_ROWS * top_k), 16);
    assert_eq!(super::marlin_block(COARSE_ROWS * top_k), 64);

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
    let capture =
        |residual_host: &[bf16], dense_host: &[bf16], scratch: &mut MoeScratch, rows: &[usize]| {
            let active_rows = residual_host.len() / hidden;
            let active_slots = active_rows * top_k;
            let residual = HiddenStates::from_host(&ctx, residual_host, hidden, active_rows)
                .expect("residual");
            let dense =
                HiddenStates::from_host(&ctx, dense_host, hidden, active_rows).expect("dense");
            let mut out = HiddenStates::zeros(&ctx, hidden, active_rows).expect("out");
            moe_into(&ctx, moe, &geom, &residual, &dense, scratch, &mut out).expect("routed block");
            let row_span = |data: &cudarc::driver::CudaSlice<bf16>, row: usize, len: usize| {
                ctx.stream
                    .clone_dtoh(&data.slice(row * len..(row + 1) * len))
                    .expect("row span")
            };
            RoutedCapture {
                padded: usize::try_from(
                    ctx.stream
                        .clone_dtoh(&scratch.padded_total)
                        .expect("padded total")[0],
                )
                .expect("padded total"),
                logits: ctx
                    .stream
                    .clone_dtoh(
                        &scratch
                            .logits
                            .data
                            .slice(..active_rows * routed.num_experts),
                    )
                    .expect("logits"),
                index: ctx
                    .stream
                    .clone_dtoh(&scratch.index.slice(..active_slots))
                    .expect("index"),
                weight: ctx
                    .stream
                    .clone_dtoh(&scratch.weight.slice(..active_slots))
                    .expect("weight"),
                rows: rows.to_vec(),
                gate: rows
                    .iter()
                    .map(|&row| (row, row_span(&scratch.routed_gate.data, row, top_k * width)))
                    .collect(),
                down: rows
                    .iter()
                    .map(|&row| {
                        (
                            row,
                            row_span(&scratch.routed_down.data, row, top_k * hidden),
                        )
                    })
                    .collect(),
                block: rows
                    .iter()
                    .map(|&row| (row, row_span(&out.data, row, hidden)))
                    .collect(),
            }
        };
    let all_rows = |rows: usize| (0..rows).collect::<Vec<_>>();
    let run = |residual_host: &[bf16], dense_host: &[bf16], scratch_rows: usize| {
        let mut scratch = MoeScratch::new(&ctx, &geom, scratch_rows).expect("scratch");
        let rows = all_rows(residual_host.len() / hidden);
        capture(residual_host, dense_host, &mut scratch, &rows)
    };

    let residual_host = sample(0, NARROW_TARGET_ROWS);
    let dense_host = sample(1, NARROW_TARGET_ROWS);
    let baseline = run(&residual_host, &dense_host, NARROW_TARGET_ROWS);
    let roomy = run(&residual_host, &dense_host, ROOMY_SCRATCH_ROWS);
    let companion_residual_host = sample(0, COMPANION_ROWS);
    let companion_dense_host = sample(1, COMPANION_ROWS);
    let companion = run(
        &companion_residual_host,
        &companion_dense_host,
        ROOMY_SCRATCH_ROWS,
    );
    assert_same_target("scratch capacity", &baseline, &roomy);
    assert_same_target("companion route", &baseline, &companion);
    assert_padding("narrow target", &baseline, routed.num_experts, 16);
    assert_padding("companion route", &companion, routed.num_experts, 16);

    // The widest 16-row dispatch and the narrowest 64-row one, each read on
    // the narrow target's four rows, a middle row and its last row.
    let edge_residual_host = sample(0, NARROW_EDGE_ROWS);
    let edge_dense_host = sample(1, NARROW_EDGE_ROWS);
    let edge_rows = [0, 1, 2, 3, NARROW_EDGE_ROWS / 2, NARROW_EDGE_ROWS - 1];
    let mut edge_scratch = MoeScratch::new(&ctx, &geom, NARROW_EDGE_ROWS).expect("edge scratch");
    let edge = capture(
        &edge_residual_host,
        &edge_dense_host,
        &mut edge_scratch,
        &edge_rows,
    );
    assert_padding("narrow edge", &edge, routed.num_experts, 16);

    let coarse_residual_host = sample(0, COARSE_ROWS);
    let coarse_dense_host = sample(1, COARSE_ROWS);
    let coarse_rows = [0, 1, 2, 3, COARSE_ROWS / 2, COARSE_ROWS - 1];
    let mut coarse_scratch = MoeScratch::new(&ctx, &geom, COARSE_ROWS).expect("coarse scratch");
    let coarse = capture(
        &coarse_residual_host,
        &coarse_dense_host,
        &mut coarse_scratch,
        &coarse_rows,
    );
    assert_padding("coarse dispatch", &coarse, routed.num_experts, 64);
    assert_prefix_gate_bytes("coarse prefix", &baseline, &coarse, top_k, width);
    let reused = capture(
        &residual_host,
        &dense_host,
        &mut coarse_scratch,
        &all_rows(NARROW_TARGET_ROWS),
    );
    assert_same_target("block 16 after block 64", &baseline, &reused);
    assert_padding("block 16 after block 64", &reused, routed.num_experts, 16);

    let host_vec = |v: &pegainfer_core::tensor::DeviceVec| -> Vec<f32> {
        ctx.stream
            .clone_dtoh(&v.data)
            .expect("norm weight")
            .iter()
            .map(|x: &bf16| x.to_f32())
            .collect()
    };
    let router_proj: Vec<f32> = ctx
        .stream
        .clone_dtoh(&moe.router_proj.data)
        .expect("router proj")
        .iter()
        .map(|x: &bf16| x.to_f32())
        .collect();

    // The reference's expert weights come from the checkpoint's bytes,
    // widened on the host rather than read in the packed form.
    let (shard_paths, _) = load_shard_info(&model).expect("shard info");
    let mmaps = mmap_shards(&shard_paths).expect("mmap shards");
    let shards = deserialize_shards(&mmaps).expect("shards");
    let plans = manifest.layers[0]
        .moe
        .as_ref()
        .expect("manifest routes layer 0");
    let host = HostBlock {
        hidden,
        width,
        top_k,
        experts: routed.num_experts,
        eps,
        router_scale: host_vec(&moe.router_scale),
        per_expert_scale: host_vec(&moe.router_per_expert_scale),
        pre_norm: host_vec(&moe.pre_feedforward_layernorm_2),
        post_dense_norm: host_vec(&moe.post_feedforward_layernorm_1),
        post_routed_norm: host_vec(&moe.post_feedforward_layernorm_2),
        router_proj,
        plans,
        shards: &shards,
        widened: std::cell::RefCell::new(std::collections::HashMap::new()),
    };
    // Every row the host oracle sees, it sees once: the narrow target's
    // four, the companion's fifth (the narrow path's tail row), the middle
    // and last rows of the widest 16-row dispatch and of the narrowest
    // 64-row one. The coarse prefix is judged by bytes above, not here.
    assert_matches_reference(
        "4-row target",
        &baseline,
        0..NARROW_TARGET_ROWS,
        &residual_host,
        &dense_host,
        &host,
    );
    assert_matches_reference(
        "companion row",
        &companion,
        NARROW_TARGET_ROWS..COMPANION_ROWS,
        &companion_residual_host,
        &companion_dense_host,
        &host,
    );
    for row in [NARROW_EDGE_ROWS / 2, NARROW_EDGE_ROWS - 1] {
        assert_matches_reference(
            "narrow edge",
            &edge,
            row..row + 1,
            &edge_residual_host,
            &edge_dense_host,
            &host,
        );
    }
    for row in [COARSE_ROWS / 2, COARSE_ROWS - 1] {
        assert_matches_reference(
            "coarse dispatch",
            &coarse,
            row..row + 1,
            &coarse_residual_host,
            &coarse_dense_host,
            &host,
        );
    }
}
