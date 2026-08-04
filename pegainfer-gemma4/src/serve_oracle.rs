//! GPU + checkpoint gates for the KV serving path. The A/B oracle needs only
//! the existing golden fixture; the generate gate needs the HF `generate()`
//! fixture dumped on the test box.

use anyhow::Result;

use super::*;
use crate::forward::full_forward;
use crate::kv::admit_tokens;
use crate::testkit::GOLDEN_PATH;
use crate::testkit::METADATA_KEY;
use crate::testkit::assert_checkpoint_matches;
use crate::testkit::i32_tensor;
use crate::testkit::model_path;

/// Compare one logit row against the oracle's: both must be finite (a NaN
/// would rank highest under `total_cmp` and be ignored by `f32::max`), the
/// argmaxes must agree, and the worst absolute gap is what the caller gates.
fn compare_row(ours: &[f32], theirs: &[f32], what: &str) -> f32 {
    assert!(
        ours.iter().chain(theirs.iter()).all(|v| v.is_finite()),
        "{what}: non-finite logit"
    );
    assert_eq!(argmax(ours), argmax(theirs), "{what}: argmax diverged");
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn fixture_manifest(bytes: &[u8], key: &str) -> serde_json::Value {
    let (_, meta) = safetensors::SafeTensors::read_metadata(bytes).expect("fixture metadata");
    serde_json::from_str(
        meta.metadata()
            .as_ref()
            .expect("fixture metadata map")
            .get(key)
            .expect("fixture manifest key"),
    )
    .expect("parse fixture manifest")
}

fn load_stack() -> (DeviceContext, GemmaServe, String) {
    let dir = model_path();
    let (weights, _) = Gemma4Weights::from_safetensors(&dir, 0).expect("load 12B weights");
    let ctx = DeviceContext::new_with_device(0).expect("device context");
    // 1024 rope rows is the window this path is fail-closed at; the pool
    // sizes cover one request at that length plus each pool's padding page.
    let serve = GemmaServe::new(&ctx, weights, 1024, 66, 66).expect("serve");
    (ctx, serve, dir)
}

/// The paged serving path against the no-KV oracle forward on the same
/// weights and tokens: each decode step is compared against a full recompute
/// of the grown sequence, so a KV write that drifts shows up immediately.
#[test]
#[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
fn serve_matches_oracle_forward() {
    let (ctx, serve, dir) = load_stack();
    let fixture_bytes = std::fs::read(GOLDEN_PATH).expect("read fixture");
    let fixture = safetensors::SafeTensors::deserialize(&fixture_bytes).expect("parse fixture");
    let (_, tokens_i32) = i32_tensor(&fixture, "short_tokens");
    let mut tokens: Vec<u32> = tokens_i32
        .iter()
        .map(|&t| u32::try_from(t).expect("token id"))
        .collect();

    // The same fixture that carries the tokens fingerprints the checkpoint
    // they were dumped from.
    assert_checkpoint_matches(&fixture_manifest(&fixture_bytes, METADATA_KEY), &dir);
    let config = &serve.weights.config;
    let local_geom = LayerGeometry::local_of(config);
    let global_geom = LayerGeometry::global_of(config);
    let (scos, ssin) = pegainfer_core::rope::precompute_rope(
        &ctx,
        &RopeTableSpec {
            rotary_dim: local_geom.head_dim,
            frequency_dim: local_geom.head_dim,
            max_seq_len: 1024,
            theta: config.sliding_rope_theta,
        },
    )
    .expect("sliding tables");
    let (gcos, gsin) = build_proportional_rope_tables(
        &ctx,
        config.global_rope_theta,
        global_geom.head_dim,
        config.global_rotary_dim,
        1024,
    )
    .expect("global tables");

    let oracle = |tokens: &[u32]| -> Vec<f32> {
        let logits = full_forward(
            &ctx,
            &serve.weights,
            tokens,
            (&scos, &ssin),
            (&gcos, &gsin),
            1024,
        )
        .expect("oracle forward");
        logits.to_host(&ctx).expect("D2H")
    };

    let mut kv = serve.alloc_kv();
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, tokens.len())
        .expect("admit prompt");
    let serve_logits = serve
        .step(&ctx, &mut kv, &tokens, LogitsSpan::All)
        .expect("serve prefill");
    let vocab = serve_logits.hidden_dim;
    let serve_host = serve_logits.to_host(&ctx).expect("D2H");
    let oracle_host = oracle(&tokens);
    let mut max_abs = 0.0f32;
    for pos in 0..tokens.len() {
        let s = &serve_host[pos * vocab..(pos + 1) * vocab];
        let o = &oracle_host[pos * vocab..(pos + 1) * vocab];
        max_abs = max_abs.max(compare_row(s, o, &format!("prefill pos {pos}")));
    }
    eprintln!(
        "prefill: {} positions, max |dlogit| {max_abs}",
        tokens.len()
    );
    // Calibrated on the box: different kernels (fused paged prep + batch
    // prefill vs two-pass contiguous + single_prefill) measured 1.31
    // peak over prefill and 0.95 over decode on softcapped logits.
    assert!(
        max_abs <= 2.0,
        "prefill |dlogit| {max_abs} above calibrated 2.0"
    );

    // Feed the ORACLE's continuation each step so one divergence cannot
    // cascade; its last row doubles as the next step's greedy pick.
    let mut oracle_last = oracle_host[(tokens.len() - 1) * vocab..tokens.len() * vocab].to_vec();
    for step in 0..4 {
        let next = u32::try_from(argmax(&oracle_last)).expect("token id");
        admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, 1).expect("admit token");
        let step_logits = serve
            .step(&ctx, &mut kv, &[next], LogitsSpan::LastRow)
            .expect("serve decode step");
        let step_host = step_logits.to_host(&ctx).expect("D2H");
        tokens.push(next);
        let o = oracle(&tokens);
        oracle_last = o[(tokens.len() - 1) * vocab..tokens.len() * vocab].to_vec();
        let s_row = &step_host[0..vocab];
        let m = compare_row(s_row, &oracle_last, &format!("decode step {step}"));
        eprintln!("decode step {step}: max |dlogit| {m}");
        assert!(
            m <= 2.0,
            "decode step {step} |dlogit| {m} above calibrated 2.0"
        );
    }
}

/// DoD gate: greedy continuation matches HF `generate()` token for
/// token on three prompts. The fixture is dumped on the box by
/// tools/accuracy/dump_gemma4_generate.py (prompt + up to 50 greedy
/// tokens per case).
#[test]
#[ignore = "requires the pinned 12B checkpoint and the generate fixture"]
fn greedy_matches_hf_generate() {
    let (ctx, serve, dir) = load_stack();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test_data/gemma4-12b-generate.safetensors"
    );
    let bytes = std::fs::read(path).expect("read generate fixture (dump on the box first)");
    // Provenance: the golden fixture fingerprints the checkpoint files, so
    // it pins what is loaded here; the generate fixture then has to name
    // that same revision, or these tokens came from another model.
    let golden_bytes = std::fs::read(GOLDEN_PATH).expect("read golden fixture");
    let golden = fixture_manifest(&golden_bytes, METADATA_KEY);
    assert_checkpoint_matches(&golden, &dir);
    let generate = fixture_manifest(&bytes, "gemma4_generate");
    assert_eq!(
        generate["revision"], golden["revision"],
        "the generate fixture was dumped from a different revision than the golden one"
    );
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let mut diverged: Vec<String> = Vec::new();
    for case in ["a", "b", "c"] {
        let (_, prompt_i32) = i32_tensor(&fixture, &format!("{case}_prompt"));
        let (_, expect_i32) = i32_tensor(&fixture, &format!("{case}_generated"));
        let prompt: Vec<u32> = prompt_i32
            .iter()
            .map(|&t| u32::try_from(t).expect("token id"))
            .collect();
        let mut kv = serve.alloc_kv();
        let ours = generate_greedy(&serve, &ctx, &mut kv, &prompt, expect_i32.len())
            .expect("greedy generation");
        let expect: Vec<u32> = expect_i32
            .iter()
            .map(|&t| u32::try_from(t).expect("token id"))
            .collect();
        assert_eq!(
            ours.len(),
            expect.len(),
            "case {case}: generated {} tokens against the fixture's {}",
            ours.len(),
            expect.len()
        );
        match ours.iter().zip(&expect).position(|(a, b)| a != b) {
            None => eprintln!("case {case}: {} tokens match HF generate", expect.len()),
            Some(at) => {
                eprintln!(
                    "case {case}: diverged at {at}/{}: ours {:?} vs HF {:?}",
                    expect.len(),
                    &ours[at..(at + 4).min(ours.len())],
                    &expect[at..(at + 4).min(expect.len())]
                );
                diverged.push(format!("{case}@{at}"));
            }
        }
    }
    assert!(
        diverged.is_empty(),
        "cases diverged from HF generate: {diverged:?}"
    );
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("non-empty row")
}

/// Greedy continuation: prefill the prompt, then decode `max_new`
/// tokens one at a time. Host argmax over the last position — the
/// correctness path; sampling belongs to the serving frontend.
fn generate_greedy(
    serve: &GemmaServe,
    ctx: &DeviceContext,
    kv: &mut GemmaKv,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>> {
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    anyhow::ensure!(max_new > 0, "generate_greedy needs max_new >= 1");
    admit_tokens(&serve.local_pool, &serve.global_pool, kv, prompt.len())?;
    let logits = serve.step(ctx, kv, prompt, LogitsSpan::LastRow)?;
    let mut next = argmax_last(ctx, &logits)?;
    let mut out = vec![next];
    for _ in 1..max_new {
        admit_tokens(&serve.local_pool, &serve.global_pool, kv, 1)?;
        let logits = serve.step(ctx, kv, &[next], LogitsSpan::LastRow)?;
        next = argmax_last(ctx, &logits)?;
        out.push(next);
    }
    Ok(out)
}

fn argmax_last(ctx: &DeviceContext, logits: &HiddenStates) -> Result<u32> {
    let host = logits.to_host(ctx)?;
    let vocab = logits.hidden_dim;
    let row = &host[(logits.seq_len - 1) * vocab..logits.seq_len * vocab];
    anyhow::ensure!(
        row.iter().all(|v| v.is_finite()),
        "non-finite logit in the row an argmax is about to rank"
    );
    let argmax = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .context("non-empty vocab")?;
    u32::try_from(argmax).context("token id fits u32")
}
