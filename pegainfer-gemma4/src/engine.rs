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
use crate::prefix_cache::PrefixCache;
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
                // Join a finished overlapped prefill: poll while other work
                // exists, block on the lane when it is the only work left.
                let lane_ready = match state.lane.as_mut() {
                    Some(lane) if lane.inflight.is_some() => {
                        let lane_is_only_work = active.is_empty() && pending.is_empty();
                        if lane_is_only_work {
                            lane.drain_or_abort();
                        }
                        lane_is_only_work || lane.inflight_complete()
                    }
                    _ => false,
                };
                if lane_ready {
                    state.join_async_prefill(&mut active);
                }
                if active.is_empty()
                    && pending.is_empty()
                    && state
                        .lane
                        .as_ref()
                        .is_none_or(|lane| lane.inflight.is_none())
                {
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
                    // With the lane busy, arrivals wait in `pending` while
                    // decode keeps stepping.
                    if state
                        .lane
                        .as_ref()
                        .is_some_and(|lane| lane.inflight.is_some())
                    {
                        break;
                    }
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
/// Overlapped admission: with the lane on, a prompt arriving into a live
/// decode batch prefills on its own stream while decode steps keep
/// replaying on `ctx.stream` — the admission costs the streams a slowdown
/// instead of a mixed step per prompt. `shared` lets the prefill grids
/// compete for every SM; `green:NN` pins the lane to NN% of them, which is
/// what actually protects decode ITL.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AsyncPrefillMode {
    Off,
    Shared,
    Green(u32),
}

fn async_prefill_mode() -> Result<AsyncPrefillMode> {
    match std::env::var("PEGAINFER_ASYNC_PREFILL") {
        Ok(raw) => parse_async_prefill_mode(&raw),
        Err(_) => Ok(AsyncPrefillMode::Off),
    }
}

fn parse_async_prefill_mode(raw: &str) -> Result<AsyncPrefillMode> {
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "0" | "false" | "off" => Ok(AsyncPrefillMode::Off),
        "shared" => Ok(AsyncPrefillMode::Shared),
        other => match other.strip_prefix("green:").and_then(|p| p.parse().ok()) {
            Some(pct) if (1..=99).contains(&pct) => Ok(AsyncPrefillMode::Green(pct)),
            _ => anyhow::bail!(
                "PEGAINFER_ASYNC_PREFILL={raw:?} not recognized (off | shared | green:NN, 1..=99)"
            ),
        },
    }
}

/// One in-flight overlapped prefill, parked until the lane's completion
/// event fires: the request, its KV, and the pass owning every device
/// buffer the in-flight kernels still read.
struct InflightPrefill {
    request: GenerateRequest,
    kv: GemmaKv,
    pass: crate::serve::PrefillPass,
    /// The cache entry this request resumed from, if any — its stale
    /// ancestor at capture time.
    resumed: Option<u64>,
}

/// The overlap lane: a dedicated prefill stream and a reusable completion
/// event. At most one prefill is in flight; while it runs, later arrivals
/// wait in the queue and decode keeps stepping — which is the point.
struct AsyncPrefillLane {
    stream: crate::green_ctx::PrefillLaneStream,
    event: cudarc::driver::CudaEvent,
    inflight: Option<InflightPrefill>,
}

impl AsyncPrefillLane {
    fn new(ctx: &DeviceContext, mode: AsyncPrefillMode) -> Result<Self> {
        let stream = match mode {
            AsyncPrefillMode::Off => anyhow::bail!("async prefill lane built with mode Off"),
            AsyncPrefillMode::Shared => crate::green_ctx::PrefillLaneStream::shared()?,
            AsyncPrefillMode::Green(pct) => {
                crate::green_ctx::PrefillLaneStream::green(ctx.device_ordinal, pct)?
            }
        };
        let event = ctx
            .ctx
            .new_event(None)
            .map_err(|e| anyhow::anyhow!("prefill completion event create failed: {e}"))?;
        Ok(Self {
            stream,
            event,
            inflight: None,
        })
    }

    /// True once the in-flight prefill's event has fired. An unexpected
    /// query error aborts: the pass's buffers may still be in use and no
    /// safe recovery exists.
    fn inflight_complete(&self) -> bool {
        debug_assert!(self.inflight.is_some());
        let query = unsafe { cudarc::driver::sys::cuEventQuery(self.event.cu_event()) };
        match query {
            cudarc::driver::sys::CUresult::CUDA_SUCCESS => true,
            cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => false,
            other => {
                log::error!("FATAL: cuEventQuery(prefill) failed ({other:?}); aborting");
                std::process::abort();
            }
        }
    }

    /// Block until the lane stream is drained; abort on failure rather
    /// than let the pass's buffers be reused under in-flight kernels.
    fn drain_or_abort(&self) {
        let sync = unsafe { cudarc::driver::sys::cuStreamSynchronize(self.stream.stream) };
        if sync != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            log::error!("FATAL: cuStreamSynchronize(prefill) failed ({sync:?}); aborting");
            std::process::abort();
        }
    }
}

impl Drop for AsyncPrefillLane {
    fn drop(&mut self) {
        self.drain_or_abort();
    }
}

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

fn send_scheduled(request: &GenerateRequest, prompt_tokens: usize, cached_tokens: usize) -> bool {
    request
        .token_tx
        .send(TokenEvent::Scheduled {
            queued_at_unix_s: request.queued_at_unix_s.unwrap_or_else(unix_now_s),
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens,
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
    /// Conversation-tail prefix cache; `None` unless
    /// `PEGAINFER_PREFIX_CACHE=K` opted in at startup.
    prefix_cache: Option<PrefixCache>,
    policy: GenerationPolicy,
    /// The value written into a suppressed slot, resident on the device so a
    /// step never stages a host scalar.
    blocked: CudaSlice<bf16>,
    base_seed: u64,
    /// Seedless sampling variety across requests comes from this counter
    /// mixed into the per-call seed; a request's own `params.seed` replays
    /// via (seed, step) regardless of it.
    sample_nonce: u64,
    /// The overlap lane; `None` unless `PEGAINFER_ASYNC_PREFILL` opted in
    /// at startup.
    lane: Option<AsyncPrefillLane>,
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
        // The cache brings its own page budget so cached entries never eat
        // serving headroom.
        let cache_entries = crate::prefix_cache::prefix_cache_cap().unwrap_or(0);
        let sliding_window = weights.config.sliding_window;
        let local_pages =
            context_pages + (MAX_CONCURRENCY - 1) * window_pages + 1 + cache_entries * window_pages;
        let global_pages = MAX_CONCURRENCY * context_pages
            + 1
            + cache_entries * crate::prefix_cache::entry_global_pages(MAX_CONTEXT);
        let serve = GemmaServe::new(&ctx, weights, MAX_CONTEXT, local_pages, global_pages)
            .map_err(|err| {
                if cache_entries > 0 {
                    err.context(format!(
                        "PEGAINFER_PREFIX_CACHE={cache_entries} grew the pools to \
                         {local_pages} local / {global_pages} global pages"
                    ))
                } else {
                    err
                }
            })?;
        let prefix_cache =
            crate::prefix_cache::prefix_cache_cap().map(|k| PrefixCache::new(k, sliding_window));
        let scratch = SampleScratch::new(&ctx, vocab, MAX_CONCURRENCY)?;
        let mut arena = serve.alloc_step_arena(&ctx, MAX_CONCURRENCY, graph_enabled)?;
        serve.precapture_decode_graphs(&ctx, &mut arena)?;
        let blocked = ctx
            .stream
            .clone_htod(&[bf16::NEG_INFINITY])
            .map_err(|err| anyhow::anyhow!("allocating the suppression sentinel failed: {err}"))?;
        let lane = match async_prefill_mode()? {
            AsyncPrefillMode::Off => None,
            mode => Some(AsyncPrefillLane::new(&ctx, mode)?),
        };
        Ok(Self {
            ctx,
            serve,
            arena,
            scratch,
            prefix_cache,
            policy,
            blocked,
            base_seed,
            sample_nonce: 0,
            lane,
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
            if send_scheduled(&request, prompt_tokens, 0) {
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

        let mut resumed = None;
        let mut kv = match self
            .prefix_cache
            .as_mut()
            .and_then(|cache| cache.resolve(&request.prompt_tokens))
        {
            Some((entry, t)) => match self.serve.restore_from_checkpoint(&self.ctx, entry, t) {
                Ok(kv) => {
                    resumed = Some(entry.id);
                    kv
                }
                Err(err) => {
                    log::warn!("gemma4 prefix-cache restore failed (falling back): {err:#}");
                    self.serve.alloc_kv()
                }
            },
            None => self.serve.alloc_kv(),
        };
        loop {
            let new_tokens = prompt_tokens - kv.local.seq_len();
            match admit_tokens(
                &self.serve.local_pool,
                &self.serve.global_pool,
                &mut kv,
                new_tokens,
            ) {
                Ok(()) => break,
                Err(err) => {
                    if self
                        .prefix_cache
                        .as_mut()
                        .is_some_and(PrefixCache::evict_lru)
                    {
                        continue;
                    }
                    if can_wait {
                        return Admitted::Requeue(Box::new((request, prefix)));
                    }
                    return reject(format!("admission refused: {err:#}"));
                }
            }
        }
        // A restored prefix is what the bridge reports as cached: the
        // resumed KV's frontier is exactly the token count served from it.
        if !send_scheduled(&request, prompt_tokens, kv.local.seq_len()) {
            return Admitted::Done;
        }

        // Overlapped admission: the prefill launches onto the lane stream
        // and this call returns immediately — decode steps continue while it
        // runs. A prompt arriving with nothing active stays on the sync
        // path: there is nothing to protect, and full-SM speed wins the head
        // of every refill burst.
        if self.lane.is_some() && !active.is_empty() {
            return self.launch_async_prefill(request, kv, resumed);
        }

        // Mixed admission: with a live decode batch, the prompt rides its
        // weight scan — one step prefills the newcomer and advances every
        // active row.
        if !active.is_empty() {
            self.ready_decode_rows(active);
            if !active.is_empty() {
                return self.mixed_admission(request, kv, active, resumed);
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
        let resume = kv.local.seq_len();
        let mut logits = match self.serve.step(
            &self.ctx,
            &mut kv,
            &request.prompt_tokens[resume..],
            LogitsSpan::LastRow,
        ) {
            Ok(logits) => logits,
            Err(err) => return fail(format!("{err:#}")),
        };
        if let Some(cache) = self.prefix_cache.as_mut() {
            if let Some(entry) =
                self.serve
                    .capture_checkpoint(&self.ctx, &kv, request.prompt_tokens.clone())
            {
                cache.insert(entry, resumed);
            }
        }
        self.first_token_flow(request, kv, &mut logits)
    }

    /// Suppress, sample and emit a prefill's first token from logits row 0,
    /// then finish the request or hand it to the decode batch — the shared
    /// tail of a sync admission and an overlapped-prefill join.
    fn first_token_flow(
        &mut self,
        request: GenerateRequest,
        kv: GemmaKv,
        logits: &mut HiddenStates,
    ) -> Admitted {
        let sink = request.token_tx.clone();
        let prompt_tokens = request.prompt_tokens.len();
        let fail = |message: String| {
            let _ = sink.send(TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens: 0,
            });
            Admitted::Done
        };
        if let Err(err) = suppress_logits(&self.ctx, &self.blocked, logits, &self.policy.suppress) {
            return fail(format!("{err:#}"));
        }

        self.sample_nonce = self.sample_nonce.wrapping_add(1);
        let call_seed = self.base_seed ^ self.sample_nonce.rotate_left(17);
        let next = match pegainfer_sample::select_batch(
            &self.ctx,
            logits,
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
                logits,
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

    /// Launch one whole-prompt prefill onto the lane stream and record the
    /// completion event. On a launch error the lane stream is drained
    /// before the KV reservation drops, so no returned page can still be
    /// written by a stale kernel.
    fn launch_async_prefill(
        &mut self,
        request: GenerateRequest,
        mut kv: GemmaKv,
        resumed: Option<u64>,
    ) -> Admitted {
        let lane = self.lane.as_mut().expect("gated by the caller");
        debug_assert!(lane.inflight.is_none());
        let launched = {
            let _guard = unsafe {
                pegainfer_core::tensor::StreamOverrideGuard::activate(lane.stream.stream)
            };
            self.serve
                .prefill_into_logits(&self.ctx, &mut kv, &request.prompt_tokens)
        };
        let recorded = launched.and_then(|pass| {
            lane.stream
                .record_event(lane.event.cu_event())
                .map(|()| pass)
        });
        match recorded {
            Ok(pass) => {
                lane.inflight = Some(InflightPrefill {
                    request,
                    kv,
                    pass,
                    resumed,
                });
                Admitted::Done
            }
            Err(err) => {
                log::error!("gemma4 async prefill launch failed: {err:#}");
                lane.drain_or_abort();
                let _ = request.token_tx.send(TokenEvent::Error {
                    message: format!("prefill failed: {err:#}"),
                    prompt_tokens: request.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                Admitted::Done
            }
        }
    }

    /// Join a completed overlapped prefill: run the deferred window
    /// release, capture into the prefix cache, and take the first-token
    /// flow the sync path uses.
    fn join_async_prefill(&mut self, active: &mut Vec<Active>) {
        let Some(lane) = self.lane.as_mut() else {
            return;
        };
        let Some(inflight) = lane.inflight.take() else {
            return;
        };
        let InflightPrefill {
            request,
            mut kv,
            mut pass,
            resumed,
        } = inflight;
        if let Err(err) = self.serve.release_prefill_window(&mut kv) {
            let _ = request.token_tx.send(TokenEvent::Error {
                message: format!("prefill window release failed: {err:#}"),
                prompt_tokens: request.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return;
        }
        if let Some(cache) = self.prefix_cache.as_mut() {
            if let Some(entry) =
                self.serve
                    .capture_checkpoint(&self.ctx, &kv, request.prompt_tokens.clone())
            {
                cache.insert(entry, resumed);
            }
        }
        if let Admitted::Active(entry) = self.first_token_flow(request, kv, &mut pass.logits) {
            active.push(*entry);
        }
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
        resumed: Option<u64>,
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
            let resume = kv.local.seq_len();
            match self.serve.mixed_prefill_decode_step(
                &self.ctx,
                &mut self.arena,
                &mut kv,
                &request.prompt_tokens[resume..],
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
        if let Some(cache) = self.prefix_cache.as_mut() {
            if let Some(entry) =
                self.serve
                    .capture_checkpoint(&self.ctx, &kv, request.prompt_tokens.clone())
            {
                cache.insert(entry, resumed);
            }
        }
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
        emit_decode_rows(active, &picked, &stops, &mut logprobs, 1);

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

        emit_decode_rows(active, &picked, &stops, &mut logprobs, 0);
    }
}

/// Deliver one decode step's outcome to every active row and retire the
/// finished ones — the event flow both the pure decode round and the mixed
/// admission share; `row_base` is the row's offset into the step's logits
/// (the mixed step's row 0 is the newcomer). A stop token retires the
/// request without being emitted; a send failure retires a cancelled one.
fn emit_decode_rows(
    active: &mut Vec<Active>,
    picked: &[u32],
    stops: &[bool],
    logprobs: &mut [Option<TokenLogprob>],
    row_base: usize,
) {
    let mut retire: Vec<usize> = Vec::new();
    for (row, entry) in active.iter_mut().enumerate() {
        if stops[row + row_base] {
            let _ = entry.request.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: entry.prompt_tokens,
                completion_tokens: entry.emitted,
            });
            retire.push(row);
            continue;
        }
        let token = picked[row + row_base];
        entry.emitted += 1;
        if entry
            .request
            .token_tx
            .send(TokenEvent::Token {
                id: token,
                logprob: logprobs[row + row_base].take(),
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

#[cfg(test)]
mod lane_tests {
    use super::AsyncPrefillMode;
    use super::parse_async_prefill_mode;

    #[test]
    fn async_prefill_mode_parses_or_refuses() {
        for off in ["", "0", "false", "off", " OFF "] {
            assert_eq!(
                parse_async_prefill_mode(off).unwrap(),
                AsyncPrefillMode::Off
            );
        }
        assert_eq!(
            parse_async_prefill_mode("shared").unwrap(),
            AsyncPrefillMode::Shared
        );
        assert_eq!(
            parse_async_prefill_mode("green:35").unwrap(),
            AsyncPrefillMode::Green(35)
        );
        for bad in [
            "1",
            "true",
            "on",
            "green",
            "green:0",
            "green:100",
            "green:x",
        ] {
            assert!(
                parse_async_prefill_mode(bad).is_err(),
                "{bad:?} must refuse, not silently degrade"
            );
        }
    }
}
