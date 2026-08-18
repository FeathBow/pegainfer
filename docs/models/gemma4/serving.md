# Gemma 4 serving

**TL;DR:** The engine schedules per iteration: up to 16 requests hold decode slots, each prompt prefills whole at a step boundary, and every active request advances one token per batched step. Prompt plus output past 8192 tokens is refused at admission, while a request that only has to wait for a decode slot queues instead. The two KV families are budgeted separately (7.27 GiB sliding + 2.00 GiB global at 12B). **This needs a 48 GiB card**: the engine sits at 32.2 GiB before it serves anything, so a 32 GiB device cannot start it. A row's output moves with the bucket widths it decodes at, but not with what its companions contain.

Last touched: 2026-08

## What a step is

The engine thread runs one loop. Each turn it admits whatever the pools can hold, up to the slot ceiling, and prefills each admitted prompt as its own step; then every active request advances exactly one token in a single batched step that shares the weight pass. A request that arrives while all slots are taken waits at the head of the queue. It is refused only when nothing is active — when there is no other request whose pages could free up, the pools genuinely cannot hold it and saying so is the honest answer.

Rows retire independently. Requests in one batch have their own frontiers, their own page tables and, for the sliding family, their own released window front, so a short request finishing does not disturb the rows that continue.

| Knob | Value | Where it binds |
| --- | --- | --- |
| Decode slots | 16 | requests beyond this queue |
| Context ceiling | 8192 tokens | prompt + `max_tokens`, enforced at admission and reported to the frontend as the servable length |
| Page size | 16 tokens | both families |
| Sliding window | 1024 tokens | the local family releases its front past this; the global family never releases |

## The two pools

Gemma 4 runs two attention families with different KV shapes, so the budget is two budgets. With 16-token pages, `C = ceil(8192/16) = 512` context pages and `W = ceil(1024/16) + 1 = 65` window pages:

```
local  = C + (slots - 1) * W + 1 = 512 + 15 * 65 + 1 = 1488 pages
global = slots * C + 1           = 16 * 512 + 1      = 8193 pages
```

The shapes behind the page: the local family is 40 layers of 8 KV heads at head_dim 256, the global family is 8 layers of 1 KV head at head_dim 512, and a page carries K and V for every layer of its family. That makes a local page 5 MiB and a global page 256 KiB, so at 12B the pools are **7.27 GiB local and 2.00 GiB global**, on top of 22.18 GiB of resident weights.

The asymmetry is the design: the local family only has to hold one full-context transient — the request currently prefilling, which has not released its front yet — on top of the window-capped steady footprint of everyone else. The global family never releases, so it stays linear in context for each request's whole lifetime, and that is what makes it the larger page count despite the smaller page.

Each pool also reserves one padding page, which is the `+ 1` in both lines.

## What a client has to send

**Prompts must carry `<bos>` themselves.** This checkpoint's `tokenizer.json` has a pass-through post-processor and its `tokenizer_config.json` sets no `add_bos_token`, so nothing in the serving path prepends it — and Gemma 4 without a leading `<bos>` degenerates into punctuation no matter how the step is scheduled:

```
prompt "The capital of France is"        -> '111.1......11111'
prompt "<bos>The capital of France is"   -> ' Paris.\nthought\nThat is correct. Paris is …'
```

The chat template in `chat_template.jinja` opens with `<bos>`, so chat-formatted prompts already carry one; see `models/gemma4/tokenizer.md` for the rest of the template contract.

Startup also logs one warning that is expected and harmless — the fast tokenizer path rejects this tokenizer's `Replace` normalizer and the server falls back to the Hugging Face tokenizers path:

```
WARN vllm_tokenizer failed to load tokenizer with fastokens; falling back to HuggingFace tokenizers
```

## Running it

```bash
cargo build --release --features gemma4 -p pegainfer-server
target/release/pegainfer \
  --model-path <checkpoint> \
  --served-model-name gemma-4-12b-it \
  --port 18099
```

```bash
curl -s localhost:18099/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b-it","prompt":"<bos>The capital of France is",
       "max_tokens":16,"temperature":0}'
```

## What it costs to hold a slot

The 16 slots are a fixed constant and the pools are sized for all of them up front. Measured on a 49140 MiB card with the default per-bucket CUDA graphs, the process sits at **33034 MiB with no request in flight** and peaked at 33386 MiB under the serving checks below: 22.18 GiB of weights, 9.27 GiB of pools, and the rest CUDA context, RoPE tables, step buffers and the captured graphs. The eager baseline (`--cuda-graph=false`) measured 32926 MiB idle and peaked at 32932 MiB.

That is a hardware floor, not a target. A 32 GiB device cannot start this configuration at all. Serving a single request needs about 2.6 GiB of pool rather than 9.27, so the slot count is what sets the floor, and it is not exposed as a knob today.

## Measured behaviour

Single GPU (sm_89, x86_64), CUDA 12.9, 12B checkpoint, greedy (`temperature 0`):

| What | Result |
| --- | --- |
| Eight distinct prompts as one batch | every row carried its own request's continuation; none carried another's |
| Eight rows asking for 4…32 tokens | each returned its own count, no row disturbed by another retiring |
| 17 concurrent requests | all completed; the ones past the slot ceiling queued rather than failing |
| One stream cancelled mid-generation | the other three finished normally |
| Four concurrent requests at 1761 prompt tokens, 200 output | all completed across the 1024-token window |

Throughput is eight prompts of 5 to 14 tokens at `max_tokens 24`, `temperature 0`, one run each, measured client-side as completion tokens over the wall time of the whole set: **31.8 tok/s** sending them one after another, **181.4 tok/s** sending them together. It is a fixed request set, not a sustained-load benchmark.

## What concurrency does and does not change

A row decoding in a batch does not produce the same logprobs as the same row decoding alone. That is worth separating from the thing it resembles — a row reading another row's pages — because only one of the two is benign. The variable that matters is the **bucket-width trajectory**: the sequence of padded bucket widths a row's decode steps actually compute at, which depends on when its companions become active and when they retire. Bucketing quantizes it — batch sizes that share a power-of-two bucket share their arithmetic — so fewer distinct trajectories exist than under exact widths. (The table below was measured at exact widths on the eager build that predates bucketing.)

| Contrast | Trajectory | Row's tokens | max abs delta logprob |
| --- | --- | --- | --- |
| One batch repeated three times | same | identical | 0.000000 |
| Companions replaced, same lengths, different content | same | identical | 0.000000 |
| Companions replaced, lengths from 2 to 1601 tokens | same | identical | 0.000000 |
| Seven short companions against seven long ones | changed | differ | 0.595163 |
| Alone against in a batch of eight | changed | differ | 0.623022 |

Hold the trajectory fixed and replace what the other rows are — their content, their prompt lengths — and the row is bit-identical, so **no companion row contaminates it**. Change when companions arrive or retire and the row moves, because the kernels pick shapes and reduction orders by batch size. (That a row reads the *right* positions and page rows in the first place is a separate question, gated by the preps' closed-form tests rather than by this comparison.)

Decode steps compute at power-of-two batch buckets — a batch pads to its bucket with rows that write the pools' reserved padding pages — and are replayed as per-bucket CUDA graphs captured at startup (`--cuda-graph=false` is the eager escape hatch; padding applies either way, so the two modes are the same arithmetic). Bucketing also quantizes the width trajectory: batch sizes that share a bucket share their arithmetic.

The consequence for callers: **greedy output is reproducible for a given workload on an otherwise idle device, not across workloads.** Replaying the same requests the same way returns the same tokens; sending them alongside different traffic changes the widths they decode at and can flip a near-tie. Another process on the same GPU does this too, by moving when each prompt's prefill lands relative to the decodes around it.

## Limits today

- **An admission rides the live decode batch instead of freezing it.** With streams in flight, a newcomer's prompt shares one eager step with the decode batch: the prompt rows sit in the row prefix, every active stream advances its token in the suffix, and one sampler call covers the newcomer's first token and every active row. Measured with a streaming request underneath a flood of one-token requests: its inter-token gap stays at about 30 ms — one mixed step — whether 16, 48 or 96 requests are queued, where the frozen-prefill scheduling this replaced measured about 500 ms at the same depths. Admission attempts per turn stay bounded by the slot ceiling; a prompt arriving with nothing active still prefills alone.
- **No chunked prefill.** A prompt runs whole in one step, so a long prompt is one long step, and admission needs its full context in pages up front.
- **No prefix cache.** Two requests sharing a prefix pay for it twice.
- **Single GPU.** No tensor parallelism for this line yet.
- **KV capacity is not reported to the frontend**: the engine logs `kv_cache_size_tokens=None`, so the frontend's capacity metrics stay empty for this model line.
