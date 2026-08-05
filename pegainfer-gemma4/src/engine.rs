//! The Gemma 4 engine: one owned thread serving requests serially through
//! the KV-backed forward. Batching, CUDA graphs and the prefix cache are
//! later stages; admission, generation, cancellation and release are real
//! from day one.

use std::path::Path;

use anyhow::Context as AnyhowContext;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::unix_now_s;
use pegainfer_sample::LogprobRequest;
use pegainfer_sample::SampleScratch;
use tokio::sync::mpsc;

use crate::forward::MULTIMODAL_PLACEHOLDER_IDS;
use crate::kv::PAGE_SIZE;
use crate::kv::admit_tokens;
use crate::serve::GemmaServe;
use crate::serve::LogitsSpan;
use crate::weights::Gemma4Weights;

/// Serving ceiling: bounds the rope tables and the pool budget. The
/// checkpoint's 262k `max_position_embeddings` needs a table and KV budget
/// design of its own.
const MAX_CONTEXT: usize = 8192;

pub(crate) fn start(model_path: &Path, options: &EngineLoadOptions) -> Result<EngineHandle> {
    let dir = model_path
        .to_str()
        .context("model path is not valid UTF-8")?
        .to_string();
    anyhow::ensure!(
        !options.enable_cuda_graph,
        "gemma4 serves eagerly; CUDA graph capture is not supported yet — \
         pass enable_cuda_graph=false"
    );
    anyhow::ensure!(
        options.device_ordinals.len() == 1,
        "gemma4 is single-device; got device_ordinals {:?}",
        options.device_ordinals
    );
    anyhow::ensure!(
        options.parallel_config.is_none(),
        "gemma4 has no parallel topology support yet"
    );
    let device = options.device_ordinals[0];
    let base_seed = options.seed;

    let policy = generation_policy(&dir)?;

    let (submit_tx, mut submit_rx) =
        mpsc::unbounded_channel::<pegainfer_frontend::engine::SubmittedRequest>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
    let join = std::thread::Builder::new()
        .name("gemma4-engine".into())
        .spawn(move || {
            let state = EngineState::load(&dir, device, policy, base_seed);
            let mut state = match state {
                Ok(state) => {
                    let _ = ready_tx.send(Ok(()));
                    state
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            while let Some((request, prefix)) = submit_rx.blocking_recv() {
                state.serve_request(&request, &prefix);
            }
        })
        .context("spawn gemma4 engine thread")?;
    ready_rx
        .recv()
        .context("gemma4 engine thread died during load")??;
    // The checkpoint advertises 262k positions this engine cannot serve.
    // Publishing the real ceiling is what lets the frontend refuse an
    // over-length request with its own message instead of forwarding one the
    // engine can only fail mid-stream.
    Ok(EngineHandle::new_with_join_handle(submit_tx, join).with_servable_len(MAX_CONTEXT as u32))
}

struct GenerationPolicy {
    eos: Vec<u32>,
    suppress: Vec<u32>,
}

fn generation_policy(dir: &str) -> Result<GenerationPolicy> {
    let path = format!("{dir}/generation_config.json");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?,
    )?;
    let eos = token_ids(
        json.get("eos_token_id")
            .with_context(|| format!("{path} missing eos_token_id"))?,
    )
    .with_context(|| format!("{path} eos_token_id"))?;
    anyhow::ensure!(!eos.is_empty(), "{path} declares an empty eos_token_id");
    let mut suppress = match json.get("suppress_tokens") {
        Some(value) => token_ids(value).with_context(|| format!("{path} suppress_tokens"))?,
        None => Vec::new(),
    };
    suppress.extend(MULTIMODAL_PLACEHOLDER_IDS);
    suppress.sort_unstable();
    suppress.dedup();
    Ok(GenerationPolicy { eos, suppress })
}

impl GenerationPolicy {
    /// Both sets index the vocabulary, so both are checked against it once
    /// the head is loaded: an out-of-range suppressed id would fail the first
    /// request, and an out-of-range stop id would never match at all and turn
    /// every request into a length stop.
    fn check_against_vocab(&self, vocab: usize) -> Result<()> {
        for (kind, ids) in [
            ("eos_token_id", &self.eos),
            ("effective suppression set", &self.suppress),
        ] {
            for &id in ids {
                anyhow::ensure!(
                    (id as usize) < vocab,
                    "{kind} lists {id}, outside the {vocab} the checkpoint's head spans"
                );
            }
        }
        Ok(())
    }
}

fn token_ids(value: &serde_json::Value) -> Result<Vec<u32>> {
    fn one(value: &serde_json::Value) -> Result<u32> {
        let raw = value
            .as_u64()
            .with_context(|| format!("{value} is not an unsigned integer"))?;
        u32::try_from(raw).with_context(|| format!("token id {raw} does not fit a u32"))
    }
    match value {
        serde_json::Value::Number(_) => Ok(vec![one(value)?]),
        serde_json::Value::Array(items) => items.iter().map(one).collect(),
        other => anyhow::bail!("unexpected token id shape: {other}"),
    }
}

/// Drop forbidden ids before sampling and logprob normalization read the row.
fn suppress_logits(
    ctx: &DeviceContext,
    blocked: &CudaSlice<bf16>,
    logits: &mut HiddenStates,
    ids: &[u32],
) -> Result<()> {
    for &id in ids {
        let index = id as usize;
        anyhow::ensure!(
            index < logits.hidden_dim,
            "suppressed token {id} is outside the {} vocabulary",
            logits.hidden_dim
        );
        for row in 0..logits.seq_len {
            let at = row * logits.hidden_dim + index;
            let mut slot = logits.data.slice_mut(at..=at);
            ctx.stream.memcpy_dtod(blocked, &mut slot)?;
        }
    }
    Ok(())
}

/// Everything the engine thread owns for the life of the process. CUDA state
/// is thread-affine, so it is built here rather than handed in: a context or
/// cuBLAS handle minted on the caller thread fails with invalid-handle on the
/// first GEMM.
struct EngineState {
    ctx: DeviceContext,
    serve: GemmaServe,
    scratch: SampleScratch,
    policy: GenerationPolicy,
    /// The value written into a suppressed slot, resident on the device so a
    /// step never stages a host scalar.
    blocked: CudaSlice<bf16>,
    base_seed: u64,
    /// Seedless sampling variety across requests comes from this counter
    /// mixed into the per-call seed; a request's own `params.seed` replays
    /// via (seed, step) regardless of it.
    sample_nonce: u64,
}

impl EngineState {
    fn load(dir: &str, device: usize, policy: GenerationPolicy, base_seed: u64) -> Result<Self> {
        let (weights, _) = Gemma4Weights::from_safetensors(dir, device)?;
        let ctx = DeviceContext::new_with_device(device)?;
        let vocab = weights.embed_tokens.rows;
        policy.check_against_vocab(vocab)?;
        // One request at a time, and a prompt is prefilled in one step: both
        // families hold every position of it before the window releases
        // anything, so each pool is sized for a full-context request plus its
        // padding page.
        let pages = MAX_CONTEXT.div_ceil(PAGE_SIZE) + 1;
        let serve = GemmaServe::new(&ctx, weights, MAX_CONTEXT, pages, pages)?;
        let scratch = SampleScratch::new(&ctx, vocab, 1)?;
        let blocked = ctx
            .stream
            .clone_htod(&[bf16::NEG_INFINITY])
            .map_err(|err| anyhow::anyhow!("allocating the suppression sentinel failed: {err}"))?;
        Ok(Self {
            ctx,
            serve,
            scratch,
            policy,
            blocked,
            base_seed,
            sample_nonce: 0,
        })
    }

    fn serve_request(
        &mut self,
        request: &GenerateRequest,
        prefix: &pegainfer_frontend::engine::KvPrefix,
    ) {
        let sink = request.token_tx.clone();
        if sink.is_closed() {
            return;
        }
        let prompt_tokens = request.prompt_tokens.len();
        let scheduled = TokenEvent::Scheduled {
            queued_at_unix_s: request.queued_at_unix_s.unwrap_or_else(unix_now_s),
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens: 0,
        };
        if sink.send(scheduled).is_err() {
            return;
        }

        let reject = |message: String| {
            let _ = sink.send(TokenEvent::Rejected {
                message,
                prompt_tokens,
                completion_tokens: 0,
            });
        };
        if prompt_tokens == 0 {
            return reject("empty prompt".into());
        }
        if request.max_tokens == 0 {
            return reject("max_tokens must be positive".into());
        }
        let context_len = prompt_tokens.checked_add(request.max_tokens);
        if context_len.is_none_or(|len| len > MAX_CONTEXT) {
            return reject(format!(
                "prompt {prompt_tokens} + max_tokens {} exceeds the serving ceiling {MAX_CONTEXT}",
                request.max_tokens
            ));
        }
        if request.lora_adapter.is_some() {
            return reject("gemma4 has no LoRA support".into());
        }
        if request.echo {
            return reject("gemma4 does not echo the prompt".into());
        }
        if prefix.hit_tokens() > 0 {
            return reject(format!(
                "gemma4 has no prefix cache yet; refusing a resolution claiming {} cached tokens",
                prefix.hit_tokens()
            ));
        }
        if request.kv_transfer_params.is_some() {
            return reject("gemma4 has no P/D transfer support; kv_transfer_params refused".into());
        }
        if let Some(rank) = request.data_parallel_rank {
            if rank != 0 {
                return reject(format!(
                    "gemma4 is single-partition; data_parallel_rank {rank} refused"
                ));
            }
        }

        match self.generate(request) {
            Ok((finish_reason, completion_tokens)) => {
                let _ = sink.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                });
            }
            Err(err) => {
                log::error!("request failed: {err:#}");
                let _ = sink.send(TokenEvent::Error {
                    message: format!("{err:#}"),
                    prompt_tokens,
                    completion_tokens: 0,
                });
            }
        }
    }

    /// Run one request to completion. Cancellation is the sink refusing an
    /// event: every emitted token checks, and the request retires mid-stream
    /// with its KV released by drop.
    fn generate(&mut self, request: &GenerateRequest) -> Result<(FinishReason, usize)> {
        let sink = &request.token_tx;
        let mut kv = self.serve.alloc_kv();

        admit_tokens(
            &self.serve.local_pool,
            &self.serve.global_pool,
            &mut kv,
            request.prompt_tokens.len(),
        )?;
        let mut logits = self.serve.step(
            &self.ctx,
            &mut kv,
            &request.prompt_tokens,
            LogitsSpan::LastRow,
        )?;

        let mut emitted = 0usize;
        loop {
            suppress_logits(&self.ctx, &self.blocked, &mut logits, &self.policy.suppress)?;
            self.sample_nonce = self.sample_nonce.wrapping_add(1);
            let call_seed = self.base_seed ^ self.sample_nonce.rotate_left(17);
            let next = pegainfer_sample::select_batch(
                &self.ctx,
                &logits,
                &[&request.params],
                &[emitted as u64],
                call_seed,
                &mut self.scratch,
            )?[0];
            // The stop token retires the request without being emitted: the
            // frontend appends its own protocol sentinel for the terminal Stop
            // and unconditionally drops the last id, so an engine that emits EOS
            // costs the client its final visible token.
            if !request.params.ignore_eos && self.policy.eos.contains(&next) {
                return Ok((FinishReason::Stop, emitted));
            }
            let logprob = if request.logprobs > 0 {
                pegainfer_sample::token_logprobs_batch(
                    &self.ctx,
                    &logits,
                    &[LogprobRequest {
                        row: 0,
                        picked: next,
                        top_k: request.logprobs,
                    }],
                )?
                .pop()
            } else {
                None
            };
            emitted += 1;
            if sink.send(TokenEvent::Token { id: next, logprob }).is_err() {
                return Ok((FinishReason::Stop, emitted));
            }
            if emitted >= request.max_tokens {
                return Ok((FinishReason::Length, emitted));
            }
            admit_tokens(&self.serve.local_pool, &self.serve.global_pool, &mut kv, 1)?;
            logits = self
                .serve
                .step(&self.ctx, &mut kv, &[next], LogitsSpan::LastRow)?;
        }
    }
}

#[cfg(test)]
mod gate {
    use super::*;

    #[test]
    fn the_suppression_mask_writes_only_the_ids_it_is_given() {
        let ctx = DeviceContext::new().expect("GPU required");
        let (vocab, rows) = (8usize, 2usize);
        let mut logits = HiddenStates::zeros(&ctx, vocab, rows).expect("logits");
        let blocked = ctx
            .stream
            .clone_htod(&[bf16::NEG_INFINITY])
            .expect("sentinel");
        suppress_logits(&ctx, &blocked, &mut logits, &[3, 5]).expect("suppress");

        let host = logits.to_host(&ctx).expect("D2H");
        for row in 0..rows {
            for id in 0..vocab {
                let value = host[row * vocab + id];
                if id == 3 || id == 5 {
                    assert!(
                        value == f32::NEG_INFINITY,
                        "row {row} id {id} is {value}, not suppressed"
                    );
                } else {
                    assert!(value == 0.0, "row {row} id {id} moved to {value}");
                }
            }
        }

        let past_the_row = suppress_logits(&ctx, &blocked, &mut logits, &[vocab as u32]);
        assert!(past_the_row.is_err(), "an id past the row must be refused");
    }
}
