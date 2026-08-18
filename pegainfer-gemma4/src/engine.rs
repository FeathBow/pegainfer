//! The Gemma 4 engine: one owned thread with iteration-level scheduling.
//! Prefill runs whole at the step boundary; every active request then
//! advances one token per batched decode step, sharing each layer's weight
//! pass.

use std::collections::VecDeque;
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
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::unix_now_s;
use pegainfer_sample::LogprobRequest;
use pegainfer_sample::SampleScratch;
use tokio::sync::mpsc;

use crate::forward::MULTIMODAL_PLACEHOLDER_IDS;
use crate::kv::GemmaKv;
use crate::kv::PAGE_SIZE;
use crate::kv::admit_tokens;
use crate::serve::GemmaServe;
use crate::serve::LogitsSpan;
use crate::serve::StepArena;
use crate::weights::Gemma4Weights;

/// Serving ceiling: bounds the rope tables and the pool budget. The
/// checkpoint's 262k `max_position_embeddings` needs a table and KV budget
/// design of its own.
const MAX_CONTEXT: usize = 8192;

/// Decode-batch ceiling: bounds the step buffers, the sampling scratch and
/// the pool budget. Admission beyond it queues at the step boundary rather
/// than rejecting.
const MAX_CONCURRENCY: usize = 16;

pub(crate) fn start(model_path: &Path, options: &EngineLoadOptions) -> Result<EngineHandle> {
    let dir = model_path
        .to_str()
        .context("model path is not valid UTF-8")?
        .to_string();
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
    let graph_enabled = options.enable_cuda_graph;

    let policy = generation_policy(&dir)?;

    let (submit_tx, mut submit_rx) =
        mpsc::unbounded_channel::<pegainfer_frontend::engine::SubmittedRequest>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
    let join = std::thread::Builder::new()
        .name("gemma4-engine".into())
        .spawn(move || {
            let state = EngineState::load(&dir, device, policy, base_seed, graph_enabled);
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
            let mut pending: VecDeque<Submitted> = VecDeque::new();
            let mut active: Vec<Active> = Vec::new();
            let mut disconnected = false;
            'engine: loop {
                loop {
                    match submit_rx.try_recv() {
                        Ok(item) => pending.push_back(item),
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }
                if active.is_empty() && pending.is_empty() {
                    if disconnected {
                        break 'engine;
                    }
                    match submit_rx.blocking_recv() {
                        Some(item) => pending.push_back(item),
                        None => break 'engine,
                    }
                }
                // A request that finishes inside its own prefill never takes
                // a slot, so slot occupancy alone does not bound this loop:
                // a backlog of one-token requests would prefill in full
                // before the next decode round and stall every stream in
                // flight for its whole length. Bound the attempts too, so a
                // burst costs the streams a bounded number of prefills per
                // token however deep it is.
                let mut attempts = 0;
                while attempts < MAX_CONCURRENCY && active.len() < MAX_CONCURRENCY {
                    let Some(item) = pending.pop_front() else {
                        break;
                    };
                    attempts += 1;
                    let can_wait = !active.is_empty();
                    match state.admit_and_prefill(item, can_wait, &mut active) {
                        Admitted::Active(request) => active.push(*request),
                        Admitted::Done => {}
                        Admitted::Requeue(item) => {
                            pending.push_front(*item);
                            break;
                        }
                    }
                }
                if !active.is_empty() {
                    state.decode_round(&mut active);
                }
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

type Submitted = pegainfer_frontend::engine::SubmittedRequest;

/// One in-flight request between decode steps: its KV, the token that feeds
/// the next step, and its progress counters.
struct Active {
    request: GenerateRequest,
    kv: GemmaKv,
    next: u32,
    emitted: usize,
    prompt_tokens: usize,
}

enum Admitted {
    Active(Box<Active>),
    /// Finished, refused, cancelled or failed: nothing carries forward.
    Done,
    /// The pools cannot hold it right now; retry once pages return.
    Requeue(Box<Submitted>),
}

fn send_scheduled(request: &GenerateRequest, prompt_tokens: usize) -> bool {
    request
        .token_tx
        .send(TokenEvent::Scheduled {
            queued_at_unix_s: request.queued_at_unix_s.unwrap_or_else(unix_now_s),
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens: 0,
        })
        .is_ok()
}

/// Everything the engine thread owns for the life of the process. CUDA state
/// is thread-affine, so it is built here rather than handed in: a context or
/// cuBLAS handle minted on the caller thread fails with invalid-handle on the
/// first GEMM.
struct EngineState {
    ctx: DeviceContext,
    serve: GemmaServe,
    arena: StepArena,
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
    fn load(
        dir: &str,
        device: usize,
        policy: GenerationPolicy,
        base_seed: u64,
        graph_enabled: bool,
    ) -> Result<Self> {
        // Refuse an unservable global GQA shape before the multi-GiB load.
        crate::serve::global_split_factor(&crate::config::Gemma4Config::from_file(dir)?)?;
        let (weights, _) = Gemma4Weights::from_safetensors(dir, device)?;
        let ctx = DeviceContext::new_with_device(device)?;
        let vocab = weights.embed_tokens.rows;
        policy.check_against_vocab(vocab)?;
        // Pool budget for a batch. A prefilling request holds every page of
        // its prompt until that step releases, so the local pool carries one
        // full-context transient on top of the window-capped steady footprint
        // of the other active requests; the global family never releases, so
        // it stays linear in context for each request's whole lifetime. Both
        // pools add the padding page they reserve.
        let context_pages = MAX_CONTEXT.div_ceil(PAGE_SIZE);
        let window_pages = weights.config.sliding_window.div_ceil(PAGE_SIZE) + 1;
        let local_pages = context_pages + (MAX_CONCURRENCY - 1) * window_pages + 1;
        let global_pages = MAX_CONCURRENCY * context_pages + 1;
        let serve = GemmaServe::new(&ctx, weights, MAX_CONTEXT, local_pages, global_pages)?;
        let scratch = SampleScratch::new(&ctx, vocab, MAX_CONCURRENCY)?;
        let mut arena = serve.alloc_step_arena(&ctx, MAX_CONCURRENCY, graph_enabled)?;
        serve.precapture_decode_graphs(&ctx, &mut arena)?;
        let blocked = ctx
            .stream
            .clone_htod(&[bf16::NEG_INFINITY])
            .map_err(|err| anyhow::anyhow!("allocating the suppression sentinel failed: {err}"))?;
        Ok(Self {
            ctx,
            serve,
            arena,
            scratch,
            policy,
            blocked,
            base_seed,
            sample_nonce: 0,
        })
    }

    /// Validate, admit and prefill one request whole, emit its first token,
    /// and hand it to the decode batch. An admission refusal is a refusal to
    /// the client only when no active request could free the pages it needs;
    /// otherwise the request waits at the queue head.
    fn admit_and_prefill(
        &mut self,
        item: Submitted,
        can_wait: bool,
        active: &mut Vec<Active>,
    ) -> Admitted {
        let (request, prefix) = item;
        let sink = request.token_tx.clone();
        if sink.is_closed() {
            return Admitted::Done;
        }
        let prompt_tokens = request.prompt_tokens.len();
        // Scheduled is paired with whatever ends the request, so a refusal
        // emits it first rather than leaving the client with no lifecycle.
        let reject = |message: String| {
            if send_scheduled(&request, prompt_tokens) {
                let _ = sink.send(TokenEvent::Rejected {
                    message,
                    prompt_tokens,
                    completion_tokens: 0,
                });
            }
            Admitted::Done
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

        let mut kv = self.serve.alloc_kv();
        if let Err(err) = admit_tokens(
            &self.serve.local_pool,
            &self.serve.global_pool,
            &mut kv,
            prompt_tokens,
        ) {
            if can_wait {
                return Admitted::Requeue(Box::new((request, prefix)));
            }
            return reject(format!("admission refused: {err:#}"));
        }
        if !send_scheduled(&request, prompt_tokens) {
            return Admitted::Done;
        }

        // Mixed admission: with a live decode batch, the prompt rides its
        // weight scan — one step prefills the newcomer and advances every
        // active row.
        if !active.is_empty() {
            self.ready_decode_rows(active);
            if !active.is_empty() {
                return self.mixed_admission(request, kv, active);
            }
        }

        let fail = |message: String| {
            let _ = sink.send(TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens: 0,
            });
            Admitted::Done
        };
        let mut logits = match self.serve.step(
            &self.ctx,
            &mut kv,
            &request.prompt_tokens,
            LogitsSpan::LastRow,
        ) {
            Ok(logits) => logits,
            Err(err) => return fail(format!("{err:#}")),
        };
        if let Err(err) =
            suppress_logits(&self.ctx, &self.blocked, &mut logits, &self.policy.suppress)
        {
            return fail(format!("{err:#}"));
        }

        self.sample_nonce = self.sample_nonce.wrapping_add(1);
        let call_seed = self.base_seed ^ self.sample_nonce.rotate_left(17);
        let next = match pegainfer_sample::select_batch(
            &self.ctx,
            &logits,
            &[&request.params],
            &[0],
            call_seed,
            &mut self.scratch,
        ) {
            Ok(tokens) => tokens[0],
            Err(err) => return fail(format!("{err:#}")),
        };
        let finish = |reason: FinishReason, completion_tokens: usize| {
            let _ = sink.send(TokenEvent::Finished {
                finish_reason: reason,
                prompt_tokens,
                completion_tokens,
            });
            Admitted::Done
        };
        // The stop token retires the request without being emitted: the
        // frontend appends its own sentinel for a terminal Stop and drops the
        // last id, so an engine that emits EOS costs the client its final
        // visible token.
        if !request.params.ignore_eos && self.policy.eos.contains(&next) {
            return finish(FinishReason::Stop, 0);
        }
        let logprob = if request.logprobs > 0 {
            match pegainfer_sample::token_logprobs_batch(
                &self.ctx,
                &logits,
                &[LogprobRequest {
                    row: 0,
                    picked: next,
                    top_k: request.logprobs,
                }],
            ) {
                Ok(mut scored) => scored.pop(),
                Err(err) => return fail(format!("{err:#}")),
            }
        } else {
            None
        };
        if sink.send(TokenEvent::Token { id: next, logprob }).is_err() {
            return Admitted::Done;
        }
        if request.max_tokens <= 1 {
            return finish(FinishReason::Length, 1);
        }
        Admitted::Active(Box::new(Active {
            request,
            kv,
            next,
            emitted: 1,
            prompt_tokens,
        }))
    }

    /// Retire every active request the next step cannot serve — a closed
    /// sink, or a pool that cannot grow its KV by one token — and admit the
    /// step's token for the rest.
    fn ready_decode_rows(&self, active: &mut Vec<Active>) {
        let mut row = 0;
        while row < active.len() {
            let entry = &mut active[row];
            if entry.request.token_tx.is_closed() {
                active.swap_remove(row);
                continue;
            }
            if let Err(err) = admit_tokens(
                &self.serve.local_pool,
                &self.serve.global_pool,
                &mut entry.kv,
                1,
            ) {
                let _ = entry.request.token_tx.send(TokenEvent::Error {
                    message: format!("{err:#}"),
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
                active.swap_remove(row);
                continue;
            }
            row += 1;
        }
    }

    /// The mixed-admission tail of [`Self::admit_and_prefill`]: the admitted
    /// prompt and the live decode batch share one step, then one sampler
    /// call covers the newcomer's first token (logits row 0) and every
    /// active row after it.
    fn mixed_admission(
        &mut self,
        request: GenerateRequest,
        mut kv: GemmaKv,
        active: &mut Vec<Active>,
    ) -> Admitted {
        let sink = request.token_tx.clone();
        let prompt_tokens = request.prompt_tokens.len();
        let fail_batch = |active: &mut Vec<Active>, what: &str, err: &anyhow::Error| {
            log::error!("{what} failed: {err:#}");
            for entry in active.drain(..) {
                let _ = entry.request.token_tx.send(TokenEvent::Error {
                    message: format!("{what} failed: {err:#}"),
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
            }
        };
        let fail = |message: String| {
            let _ = sink.send(TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens: 0,
            });
            Admitted::Done
        };

        let decode_tokens: Vec<u32> = active.iter().map(|entry| entry.next).collect();
        let logits = {
            let mut kvs: Vec<&mut GemmaKv> = active.iter_mut().map(|entry| &mut entry.kv).collect();
            match self.serve.mixed_prefill_decode_step(
                &self.ctx,
                &mut self.arena,
                &mut kv,
                &request.prompt_tokens,
                &mut kvs,
                &decode_tokens,
            ) {
                Ok(logits) => logits,
                Err(err) => {
                    fail_batch(active, "mixed step", &err);
                    return fail(format!("mixed step failed: {err:#}"));
                }
            }
        };
        if let Err(err) = suppress_logits(&self.ctx, &self.blocked, logits, &self.policy.suppress) {
            fail_batch(active, "mixed suppression", &err);
            return fail(format!("mixed suppression failed: {err:#}"));
        }

        self.sample_nonce = self.sample_nonce.wrapping_add(1);
        let call_seed = self.base_seed ^ self.sample_nonce.rotate_left(17);
        let picked = {
            let params: Vec<_> = std::iter::once(&request.params)
                .chain(active.iter().map(|entry| &entry.request.params))
                .collect();
            let steps: Vec<u64> = std::iter::once(0u64)
                .chain(active.iter().map(|entry| entry.emitted as u64))
                .collect();
            match pegainfer_sample::select_batch(
                &self.ctx,
                logits,
                &params,
                &steps,
                call_seed,
                &mut self.scratch,
            ) {
                Ok(picked) => picked,
                Err(err) => {
                    fail_batch(active, "mixed sampling", &err);
                    return fail(format!("mixed sampling failed: {err:#}"));
                }
            }
        };
        let mut stops = vec![false; active.len() + 1];
        stops[0] = !request.params.ignore_eos && self.policy.eos.contains(&picked[0]);
        for (row, entry) in active.iter().enumerate() {
            stops[row + 1] =
                !entry.request.params.ignore_eos && self.policy.eos.contains(&picked[row + 1]);
        }
        let mut lp_requests: Vec<LogprobRequest> = Vec::new();
        if request.logprobs > 0 && !stops[0] {
            lp_requests.push(LogprobRequest {
                row: 0,
                picked: picked[0],
                top_k: request.logprobs,
            });
        }
        lp_requests.extend(
            active
                .iter()
                .enumerate()
                .filter(|(row, entry)| entry.request.logprobs > 0 && !stops[row + 1])
                .map(|(row, entry)| LogprobRequest {
                    row: row + 1,
                    picked: picked[row + 1],
                    top_k: entry.request.logprobs,
                }),
        );
        let mut logprobs: Vec<Option<TokenLogprob>> = vec![None; active.len() + 1];
        if !lp_requests.is_empty() {
            match pegainfer_sample::token_logprobs_batch(&self.ctx, logits, &lp_requests) {
                Ok(scored) => {
                    for (lp, scored) in lp_requests.iter().zip(scored) {
                        logprobs[lp.row] = Some(scored);
                    }
                }
                Err(err) => {
                    fail_batch(active, "mixed logprobs", &err);
                    return fail(format!("mixed logprobs failed: {err:#}"));
                }
            }
        }

        // Active rows: the decode-round event flow, one logits row up.
        let mut retire: Vec<usize> = Vec::new();
        for (row, entry) in active.iter_mut().enumerate() {
            if stops[row + 1] {
                let _ = entry.request.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
                retire.push(row);
                continue;
            }
            let token = picked[row + 1];
            entry.emitted += 1;
            if entry
                .request
                .token_tx
                .send(TokenEvent::Token {
                    id: token,
                    logprob: logprobs[row + 1].take(),
                })
                .is_err()
            {
                retire.push(row);
                continue;
            }
            if entry.emitted >= entry.request.max_tokens {
                let _ = entry.request.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Length,
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
                retire.push(row);
                continue;
            }
            entry.next = token;
        }
        for row in retire.into_iter().rev() {
            active.swap_remove(row);
        }

        // The newcomer: its first token is logits row 0.
        let finish = |reason: FinishReason, completion_tokens: usize| {
            let _ = sink.send(TokenEvent::Finished {
                finish_reason: reason,
                prompt_tokens,
                completion_tokens,
            });
            Admitted::Done
        };
        if stops[0] {
            return finish(FinishReason::Stop, 0);
        }
        let next = picked[0];
        if sink
            .send(TokenEvent::Token {
                id: next,
                logprob: logprobs[0].take(),
            })
            .is_err()
        {
            return Admitted::Done;
        }
        if request.max_tokens <= 1 {
            return finish(FinishReason::Length, 1);
        }
        Admitted::Active(Box::new(Active {
            request,
            kv,
            next,
            emitted: 1,
            prompt_tokens,
        }))
    }

    /// One batched decode step: every active request advances a token,
    /// sharing each layer's weight pass. A cancelled request and a request
    /// the pools cannot grow for retire before the batch is built; a finished
    /// one retires after its token lands.
    fn decode_round(&mut self, active: &mut Vec<Active>) {
        self.ready_decode_rows(active);
        if active.is_empty() {
            return;
        }

        let fail_batch = |active: &mut Vec<Active>, what: &str, err: &anyhow::Error| {
            log::error!("{what} failed: {err:#}");
            for entry in active.drain(..) {
                let _ = entry.request.token_tx.send(TokenEvent::Error {
                    message: format!("{what} failed: {err:#}"),
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
            }
        };

        let tokens: Vec<u32> = active.iter().map(|entry| entry.next).collect();
        let logits = {
            let mut kvs: Vec<&mut GemmaKv> = active.iter_mut().map(|entry| &mut entry.kv).collect();
            match self
                .serve
                .decode_batch_step(&self.ctx, &mut self.arena, &mut kvs, &tokens)
            {
                Ok(logits) => logits,
                Err(err) => return fail_batch(active, "batched decode", &err),
            }
        };
        if let Err(err) = suppress_logits(&self.ctx, &self.blocked, logits, &self.policy.suppress) {
            return fail_batch(active, "suppression", &err);
        }

        self.sample_nonce = self.sample_nonce.wrapping_add(1);
        let call_seed = self.base_seed ^ self.sample_nonce.rotate_left(17);
        let picked = {
            let params: Vec<_> = active.iter().map(|entry| &entry.request.params).collect();
            let steps: Vec<u64> = active.iter().map(|entry| entry.emitted as u64).collect();
            match pegainfer_sample::select_batch(
                &self.ctx,
                logits,
                &params,
                &steps,
                call_seed,
                &mut self.scratch,
            ) {
                Ok(picked) => picked,
                Err(err) => return fail_batch(active, "batched sampling", &err),
            }
        };
        let mut stops = vec![false; active.len()];
        for (row, entry) in active.iter().enumerate() {
            stops[row] = !entry.request.params.ignore_eos && self.policy.eos.contains(&picked[row]);
        }
        let lp_requests: Vec<LogprobRequest> = active
            .iter()
            .enumerate()
            .filter(|(row, entry)| entry.request.logprobs > 0 && !stops[*row])
            .map(|(row, entry)| LogprobRequest {
                row,
                picked: picked[row],
                top_k: entry.request.logprobs,
            })
            .collect();
        let mut logprobs: Vec<Option<TokenLogprob>> = vec![None; active.len()];
        if !lp_requests.is_empty() {
            match pegainfer_sample::token_logprobs_batch(&self.ctx, logits, &lp_requests) {
                Ok(scored) => {
                    for (request, logprob) in lp_requests.iter().zip(scored) {
                        logprobs[request.row] = Some(logprob);
                    }
                }
                Err(err) => return fail_batch(active, "batched logprobs", &err),
            }
        }

        let mut retire: Vec<usize> = Vec::new();
        for (row, entry) in active.iter_mut().enumerate() {
            if stops[row] {
                let _ = entry.request.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
                retire.push(row);
                continue;
            }
            let token = picked[row];
            entry.emitted += 1;
            if entry
                .request
                .token_tx
                .send(TokenEvent::Token {
                    id: token,
                    logprob: logprobs[row].take(),
                })
                .is_err()
            {
                retire.push(row);
                continue;
            }
            if entry.emitted >= entry.request.max_tokens {
                let _ = entry.request.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Length,
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
                retire.push(row);
                continue;
            }
            entry.next = token;
        }
        for row in retire.into_iter().rev() {
            active.swap_remove(row);
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
