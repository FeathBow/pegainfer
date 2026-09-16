"""TileLang definition of Gemma 4's global-attention prefill at head dim 512.

Authored here, not vendored: upstream has no kernel for this head dim on
SM90 — FlashInfer's Hopper prefill compiles 64, 128 and 256 only — so the
serving path runs a generic paged kernel that leaves most of the machine
idle at this shape.

Three things make this faster, and none of them is the loop body:

  * the grid walks a KV group's query heads across a block of query tiles
    before advancing, so the CTAs that want the same key tile stay resident
    together and step through the context in lockstep;
  * the score tile goes from shared memory straight into the value gemm,
    because copying it back into a fragment first costs a fifth of the
    runtime for a bit-identical result; and
  * one key tile is exactly one page, so its load is a single unwrapped
    copy — wrapping it in a loop, or splitting it into per-page pieces,
    forfeits the bulk path and with it about half the throughput.

The last one is why the global KV family pages at the key block rather than
at the sliding family's finer granularity.

Ragged batches are the serving case: a mixed step launches its prompt rows
as one plan with an entry per segment. The kernel adds no array of its own:
it walks the batch once per CTA to find which request owns it, summing each
request's CTA count with the tile count rounded up to a whole mapping block
so the group walk stays intact. That is a handful of shifts over a bounded
batch, and it keeps the inputs to exactly what the plan already carries. A
request that is not in the step contributes no CTAs, so it is never chosen,
and a CTA past its request's real tiles exits before it touches memory.

Each request's context length comes the same way, from how far its slice of
the page table reaches and how full its last page is, rather than as another
array restating what those already say.

Rows are packed, so the store is predicated: a partial last tile would
otherwise write over the next request's rows, which in a mixed step are the
decode rows sharing the output buffer. Keeping the bulk copy for whole tiles
and branching to the predicated store only for a request's last one measures
slower than predicating every tile, so there is no branch.
"""

import tilelang
import tilelang.language as T

DTYPE = "bfloat16"
ACC = "float"

# The fp32 output accumulator is block_M * head_dim * 4 B, so block_M is 64:
# 128 spills half the register file. block_N is the page size, and the two
# gemms want a full warpgroup pair.
BLOCK_M = 64
BLOCK_N = 64
NUM_STAGES = 1
THREADS = 256

# Query tiles per KV group before the grid advances.
QBLK = 8

PASS_CONFIGS = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


def prefill_varlen(
    heads,
    groups,
    dim,
    page_size,
    max_batch,
    q_rows,
    pool_rows,
    page_table_len,
    block_M=BLOCK_M,
    block_N=BLOCK_N,
    num_stages=NUM_STAGES,
    threads=THREADS,
    qblk=QBLK,
):
    """Causal GQA prefill over the paged pool, ragged across requests.

    `q_rows`, `pool_rows` and `page_table_len` are declared at the serving
    arena's maxima. They reach the generated code only as bounds guards, so a
    step smaller than the arena passes them all; the real bounds come from the
    store predicate and from the walk's own trip count. The two tensors the
    lowering reads through TMA carry their extents in the descriptors the
    launcher builds, so those are the step's own.
    """
    assert block_N == page_size, "one tile must be one page, or the load splits"
    head_kv = heads // groups
    q_shape = [q_rows, heads, dim]
    kv_shape = [pool_rows, head_kv, dim]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, DTYPE),
        KV: T.Tensor(kv_shape, DTYPE),
        PageIndices: T.Tensor([page_table_len], "int32"),
        PageIndptr: T.Tensor([max_batch + 1], "int32"),
        QIndptr: T.Tensor([max_batch + 1], "int32"),
        LastPageLen: T.Tensor([max_batch], "int32"),
        sm_scale: T.float32,
        total_ctas: T.int32,
        rows_per_page: T.int32,
        layer_row: T.int32,
        Output: T.Tensor(q_shape, DTYPE),
    ):
        with T.Kernel(total_ctas, threads=threads) as pid:
            # The softmax runs on exp2, so the caller's scale carries log2(e)
            # into the exponent. It is the caller's because the serving path
            # folds 1/sqrt(head_dim) into the query rows upstream and hands
            # this kernel a scale of one; baking the usual factor in here
            # would apply it twice and quietly flatten every distribution.
            scale = sm_scale * 1.44269504
            Q_shared = T.alloc_shared([block_M, dim], DTYPE)
            K_shared = T.alloc_shared([block_N, dim], DTYPE)
            S_shared = T.alloc_shared([block_M, block_N], DTYPE)
            V_shared = T.alloc_shared([block_N, dim], DTYPE)
            acc_s = T.alloc_fragment([block_M, block_N], ACC)
            acc_o = T.alloc_fragment([block_M, dim], ACC)
            scores_max = T.alloc_fragment([block_M], ACC)
            scores_max_prev = T.alloc_fragment([block_M], ACC)
            scores_scale = T.alloc_fragment([block_M], ACC)
            scores_sum = T.alloc_fragment([block_M], ACC)
            logsum = T.alloc_fragment([block_M], ACC)
            req = T.alloc_local([1], "int32")
            first_cta = T.alloc_local([1], "int32")
            own_ctas = T.alloc_local([1], "int32")
            walked = T.alloc_local([1], "int32")

            # Which request owns this CTA, and where its block starts. The
            # host sizes the grid with the same sum, so the two agree by
            # construction rather than through an array that could drift.
            req[0] = 0
            first_cta[0] = 0
            own_ctas[0] = 0
            walked[0] = 0
            # `total_ctas` is the host's same sum, so it also says how many
            # boundaries are real: the caller's array is as long as its own
            # batch, while `max_batch` is this kernel's ceiling.
            for i in T.serial(max_batch):
                if walked[0] < total_ctas:
                    mine = (
                        T.ceildiv(T.ceildiv(QIndptr[i + 1] - QIndptr[i], block_M), qblk)
                        * qblk
                        * heads
                    )
                    # Requests sit back to back, so the owner is the last one
                    # that both starts at or before this CTA and has any. The
                    # emptiness test is what keeps a request the step left out
                    # from claiming CTAs it has no tiles for.
                    if walked[0] <= pid and mine > 0:
                        req[0] = i
                        first_cta[0] = walked[0]
                        own_ctas[0] = mine
                    walked[0] = walked[0] + mine
            # An over-sized grid then does nothing rather than recompute
            # somebody else's tile; an under-sized one leaves a tail, which
            # the numerics gate sees.
            if pid < walked[0]:
                b = req[0]
                local = pid - first_cta[0]
                q_tiles = own_ctas[0] // heads
                per_group = q_tiles * groups
                r = local % per_group
                head = local // per_group * groups + (r % (qblk * groups)) // qblk
                q_tile = r // (qblk * groups) * qblk + r % qblk
                kv_head = head // groups
                q_start = QIndptr[b]
                q_len = QIndptr[b + 1] - q_start
                pages = PageIndptr[b + 1] - PageIndptr[b]
                kv_len = (pages - 1) * page_size + LastPageLen[b]
                offset = kv_len - q_len
                row = q_start + q_tile * block_M

                if q_tile * block_M < q_len:
                    T.copy(Q[row : row + block_M, head, :], Q_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(ACC))

                    loop_range = T.min(
                        T.ceildiv(offset + (q_tile + 1) * block_M, block_N),
                        T.ceildiv(kv_len, block_N),
                    )
                    for k in T.Pipelined(loop_range, num_stages=num_stages):
                        # One page holds every layer's K then V for `page_size`
                        # tokens, so the layer's K block starts `layer_row` into
                        # the page and its V block one page further.
                        kb = PageIndices[PageIndptr[b] + k] * rows_per_page + layer_row
                        T.copy(KV[kb : kb + page_size, kv_head, :], K_shared)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                q_tile * block_M + i + offset < k * block_N + j, -1e9, 0
                            )
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )

                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(ACC))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        for i in T.Parallel(block_M):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] * scale - scores_max[i] * scale
                            )
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.exp2(
                                acc_s[i, j] * scale - scores_max[i] * scale
                            )
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_M):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, S_shared)

                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(
                            KV[kb + page_size : kb + 2 * page_size, kv_head, :],
                            V_shared,
                        )
                        T.gemm(
                            S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol
                        )

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = acc_o[i, j] / logsum[i]

                    # Q_shared is dead once the walk ends, and the four live shared
                    # buffers already sit at the SM90 dynamic limit, so the store
                    # stages through it rather than its own.
                    T.copy(acc_o, Q_shared)
                    for i, d in T.Parallel(block_M, dim):
                        if q_tile * block_M + i < q_len:
                            Output[row + i, head, d] = Q_shared[i, d]

    return main
