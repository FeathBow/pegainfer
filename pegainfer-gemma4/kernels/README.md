# Gemma 4 TileLang kernels

**TL;DR**: `generate.py` AOT-compiles the hd512 global-attention prefill in
`tilelang_defs.py` into one CUDA file that `pegainfer-kernels/build.rs` hands
to nvcc under the `gemma4` feature — three tiers (generate, pre-generated,
stub), and the generated CUDA is a Cargo `OUT_DIR` artifact that is never
checked in. Unlike the K3 families this one lowers to TMA, so the launcher
builds the descriptors itself from parameters recovered out of the lowered
host stub.

## What lives here

| File | Role |
| --- | --- |
| `tilelang_defs.py` | The kernel, authored here. Upstream has nothing for this head dim on SM90. |
| `generate.py` | Lowers it, recovers the launch geometry and the TMA descriptor parameters, emits the `.cu` with a hand-written launcher. |

## Why one instantiation is enough

Every shape a TileLang kernel declares is a compile dimension, and a serving
step's packed query rows, page-table length and pool size are all run-time
quantities. They are nevertheless a single instantiation, because the declared
extents reach the generated code only as bounds guards — two lowerings that
differ solely in them differ in nothing else — so declaring the serving
arena's maxima lets every smaller step through. The real bounds are the walk's
own trip count and the predicated store.

The two tensors the lowering reads through TMA are separate: their extents
live in the descriptors, which the launcher builds per call from the arguments
it is given, so those are always the step's own.

## The descriptor parameters are recovered, not written down

TileLang exposes no accessor for the launch it baked into the host stub, so
`generate.py` parses the packed-call argument stack — the same technique the
K3 generator uses for its launch geometry, extended to the descriptor builds.
Each parameter is then mapped to its driver enum through a table with no
default, and the tensor and descriptor names are bound through tables with no
default either, so a codegen change fails generation instead of silently
encoding a stale descriptor.

The K3 generator refuses a TMA-lowered body outright, for exactly the reason
this file exists: its launchers bind plain pointers and the requested thread
count, and a warp-specialized kernel accepts neither.

## Gates

The kernel's numerics are gated against the serving path's own reference, not
against random tensors: paged output is bit-identical to the contiguous form
over scattered pages and partial final pages, and a ragged batch matches an
fp32 reference per request while leaving every row past the batch untouched —
those rows are the decode rows sharing a mixed step's output buffer.
