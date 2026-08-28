// Gemma 4 routed-expert dispatch.
//
// The router's softmax runs over every expert, so one block owns one token
// and keeps the whole expert row in shared memory. The scatter is launched
// once per expert, which is what makes it atomic-free: within one launch no
// two slots name the same destination token.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

namespace {

// One warp per token: every reduction below is a shuffle, so a block wider
// than a warp would silently drop the other warps' partials.
constexpr int kRouterBlock = 32;

__global__ void gemma4_moe_router_topk_kernel(
    const __nv_bfloat16 *__restrict__ logits,
    const __nv_bfloat16 *__restrict__ per_expert_scale, int experts, int top_k,
    int *__restrict__ index_out, float *__restrict__ weight_out) {
  extern __shared__ float probability[];
  const int row = blockIdx.x;
  const __nv_bfloat16 *row_logits = logits + (long long)row * experts;

  __shared__ float reduced;
  __shared__ int reduced_at;

  float mine = -FLT_MAX;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    float value = __bfloat162float(row_logits[e]);
    probability[e] = value;
    mine = fmaxf(mine, value);
  }
  for (int width = warpSize / 2; width > 0; width >>= 1) {
    mine = fmaxf(mine, __shfl_down_sync(0xffffffff, mine, width));
  }
  if (threadIdx.x == 0) {
    reduced = mine;
  }
  __syncthreads();

  const float peak = reduced;
  float total = 0.0f;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    float value = __expf(probability[e] - peak);
    probability[e] = value;
    total += value;
  }
  for (int width = warpSize / 2; width > 0; width >>= 1) {
    total += __shfl_down_sync(0xffffffff, total, width);
  }
  if (threadIdx.x == 0) {
    reduced = total;
  }
  __syncthreads();

  const float norm = reduced;
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    probability[e] /= norm;
  }
  __syncthreads();

  // Selection is serial in k because each pick has to mask the previous one.
  // A tie takes the lower expert, which is what torch.topk reports.
  float selected = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    float best = -FLT_MAX;
    int best_at = experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
      float value = probability[e];
      if (value > best || (value == best && e < best_at)) {
        best = value;
        best_at = e;
      }
    }
    for (int width = warpSize / 2; width > 0; width >>= 1) {
      float other = __shfl_down_sync(0xffffffff, best, width);
      int other_at = __shfl_down_sync(0xffffffff, best_at, width);
      if (other > best || (other == best && other_at < best_at)) {
        best = other;
        best_at = other_at;
      }
    }
    if (threadIdx.x == 0) {
      reduced = best;
      reduced_at = best_at;
    }
    __syncthreads();
    const int taken = reduced_at;
    if (threadIdx.x == 0) {
      index_out[(long long)row * top_k + k] = taken;
      weight_out[(long long)row * top_k + k] = reduced;
      probability[taken] = -FLT_MAX;
    }
    selected += reduced;
    __syncthreads();
  }

  // The selected probabilities are renormalized among themselves before the
  // per-expert scale lands, so a token's expert weights sum to one first.
  if (threadIdx.x == 0) {
    for (int k = 0; k < top_k; ++k) {
      const long long at = (long long)row * top_k + k;
      const int expert = index_out[at];
      weight_out[at] =
          weight_out[at] / selected * __bfloat162float(per_expert_scale[expert]);
    }
  }
}

__global__ void gemma4_moe_scatter_add_kernel(
    const __nv_bfloat16 *__restrict__ delta, const int *__restrict__ rows,
    const float *__restrict__ weights, int hidden, long long total,
    __nv_bfloat16 *__restrict__ out) {
  long long at = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (at >= total) {
    return;
  }
  const int slot = (int)(at / hidden);
  const int column = (int)(at % hidden);
  const long long destination = (long long)rows[slot] * hidden + column;
  const float scaled =
      __bfloat162float(delta[at]) * weights[slot] + __bfloat162float(out[destination]);
  out[destination] = __float2bfloat16(scaled);
}


__global__ void gemma4_moe_sum_topk_kernel(
    const __nv_bfloat16 *__restrict__ routed, int top_k, int hidden,
    long long total, __nv_bfloat16 *__restrict__ out) {
  long long at = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (at >= total) {
    return;
  }
  const long long token = at / hidden;
  const int column = (int)(at % hidden);
  float sum = 0.0f;
  for (int pick = 0; pick < top_k; ++pick) {
    sum += __bfloat162float(routed[(token * top_k + pick) * hidden + column]);
  }
  out[at] = __float2bfloat16(sum);
}

} // namespace

extern "C" {

// `logits` is `[rows, experts]` bf16 as the router projection leaves it.
// `index_out` and `weight_out` are `[rows, top_k]`.
CUresult gemma4_moe_router_topk_cuda(const __nv_bfloat16 *logits,
                                     const __nv_bfloat16 *per_expert_scale,
                                     int rows, int experts, int top_k,
                                     int *index_out, float *weight_out,
                                     cudaStream_t stream) {
  if (logits == nullptr || per_expert_scale == nullptr ||
      index_out == nullptr || weight_out == nullptr || rows <= 0 ||
      experts <= 0 || top_k <= 0 || top_k > experts) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t shared = (size_t)experts * sizeof(float);
  gemma4_moe_router_topk_kernel<<<rows, kRouterBlock, shared, stream>>>(
      logits, per_expert_scale, experts, top_k, index_out, weight_out);
  return (CUresult)cudaGetLastError();
}

// `out[rows[slot]] += weights[slot] * delta[slot]`. One launch per expert:
// the destinations within a launch are distinct, so no atomics.
CUresult gemma4_moe_scatter_add_cuda(const __nv_bfloat16 *delta,
                                     const int *rows, const float *weights,
                                     int slots, int hidden,
                                     __nv_bfloat16 *out, cudaStream_t stream) {
  if (delta == nullptr || rows == nullptr || weights == nullptr ||
      out == nullptr || slots <= 0 || hidden <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const long long total = (long long)slots * hidden;
  const int block = 256;
  const long long grid = (total + block - 1) / block;
  if (grid > 2147483647LL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  gemma4_moe_scatter_add_kernel<<<(int)grid, block, 0, stream>>>(
      delta, rows, weights, hidden, total, out);
  return (CUresult)cudaGetLastError();
}


// The routed GEMM leaves one row per (token, pick); this folds the picks back
// onto their token. The per-pick weights are already in the rows.
CUresult gemma4_moe_sum_topk_cuda(const __nv_bfloat16 *routed, int rows,
                                  int top_k, int hidden, __nv_bfloat16 *out,
                                  cudaStream_t stream) {
  if (routed == nullptr || out == nullptr || rows <= 0 || top_k <= 0 ||
      hidden <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const long long total = (long long)rows * hidden;
  const int block = 256;
  const long long grid = (total + block - 1) / block;
  if (grid > 2147483647LL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  gemma4_moe_sum_topk_kernel<<<(int)grid, block, 0, stream>>>(
      routed, top_k, hidden, total, out);
  return (CUresult)cudaGetLastError();
}

} // extern "C"
