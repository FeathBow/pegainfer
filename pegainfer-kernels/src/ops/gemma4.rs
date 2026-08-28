use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;

/// Softmax the router logits over every expert, take the top `top_k`,
/// renormalize those among themselves, and apply the per-expert scale.
pub fn gemma4_moe_router_topk_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    per_expert_scale: &DeviceVec,
    top_k: usize,
    index_out: &mut CudaSlice<i32>,
    weight_out: &mut CudaSlice<f32>,
) -> Result<()> {
    let rows = logits.seq_len;
    let experts = logits.hidden_dim;
    ensure!(
        rows > 0 && top_k > 0 && top_k <= experts,
        "gemma4_moe_router_topk_into: {rows} rows and top {top_k} of {experts} experts is not \
         a routing problem"
    );
    ensure!(
        per_expert_scale.len == experts,
        "gemma4_moe_router_topk_into: the per-expert scale holds {}, not {experts}",
        per_expert_scale.len
    );
    let slots = rows
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("gemma4_moe_router_topk_into: {rows} x {top_k} overflows usize"))?;
    ensure!(
        logits.data.len() >= rows * experts
            && index_out.len() >= slots
            && weight_out.len() >= slots,
        "gemma4_moe_router_topk_into buffers too small for {rows} x {top_k}: logits {}, index {}, \
         weight {}",
        logits.data.len(),
        index_out.len(),
        weight_out.len()
    );
    let rows_i32 = i32::try_from(rows)
        .map_err(|_| anyhow!("gemma4_moe_router_topk_into: {rows} rows exceed the kernel's i32"))?;
    let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = per_expert_scale.data.device_ptr(&ctx.stream);
    let (index_ptr, _index_guard) = index_out.device_ptr_mut(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight_out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gemma4_moe_router_topk_cuda(
            logits_ptr as *const ffi::Half,
            scale_ptr as *const ffi::Half,
            rows_i32,
            experts as i32,
            top_k as i32,
            index_ptr as *mut i32,
            weight_ptr as *mut f32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Fold a routed GEMM's `[rows * top_k, hidden]` result back onto its tokens.
pub fn gemma4_moe_sum_topk_into(
    ctx: &DeviceContext,
    routed: &HiddenStates,
    top_k: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    let rows = out.seq_len;
    let hidden = out.hidden_dim;
    ensure!(
        rows > 0 && top_k > 0 && routed.hidden_dim == hidden,
        "gemma4_moe_sum_topk_into: routed is {} wide, out is {hidden}",
        routed.hidden_dim
    );
    ensure!(
        routed.seq_len >= rows * top_k,
        "gemma4_moe_sum_topk_into: routed holds {} rows, not {}",
        routed.seq_len,
        rows * top_k
    );
    let (routed_ptr, _routed_guard) = routed.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gemma4_moe_sum_topk_cuda(
            routed_ptr as *const ffi::Half,
            i32::try_from(rows)?,
            i32::try_from(top_k)?,
            i32::try_from(hidden)?,
            out_ptr as *mut ffi::Half,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Everything one Marlin NVFP4 GEMM needs beyond its operands.
pub struct MarlinDispatch<'a> {
    pub sorted_token_ids: &'a CudaSlice<i32>,
    pub expert_ids: &'a CudaSlice<i32>,
    pub num_tokens_post_padded: &'a CudaSlice<i32>,
    pub topk_weights: &'a CudaSlice<f32>,
    pub block_size: usize,
    /// `top_k` for the first projection, which gathers each token once per
    /// pick; `1` for the second, whose rows are already per pick.
    pub top_k: usize,
    pub mul_topk_weights: bool,
}

/// One expert-blocked NVFP4 GEMM. `rows` is the A operand's row count, which
/// the dispatch's `top_k` expands to the output's.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_marlin_nvfp4_moe(
    ctx: &DeviceContext,
    input: &HiddenStates,
    qweight: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    global_scale: &CudaSlice<f32>,
    workspace: &mut CudaSlice<i32>,
    c_tmp: &mut CudaSlice<f32>,
    dispatch: &MarlinDispatch<'_>,
    rows: usize,
    size_n: usize,
    size_k: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        rows > 0 && size_n > 0 && size_k > 0,
        "gemma4_marlin_nvfp4_moe: {rows} x {size_n} x {size_k} is not a GEMM"
    );
    ensure!(
        input.hidden_dim == size_k && input.seq_len >= rows,
        "gemma4_marlin_nvfp4_moe: input is {} x {}, needs {rows} x {size_k}",
        input.seq_len,
        input.hidden_dim
    );
    ensure!(
        out.hidden_dim == size_n && out.seq_len >= rows * dispatch.top_k,
        "gemma4_marlin_nvfp4_moe: out is {} x {}, needs {} x {size_n}",
        out.seq_len,
        out.hidden_dim,
        rows * dispatch.top_k
    );
    let (input_ptr, _input_guard) = input.data.device_ptr(&ctx.stream);
    let (qweight_ptr, _qweight_guard) = qweight.device_ptr(&ctx.stream);
    let (scales_ptr, _scales_guard) = scales.device_ptr(&ctx.stream);
    let (global_ptr, _global_guard) = global_scale.device_ptr(&ctx.stream);
    let (sorted_ptr, _sorted_guard) = dispatch.sorted_token_ids.device_ptr(&ctx.stream);
    let (expert_ptr, _expert_guard) = dispatch.expert_ids.device_ptr(&ctx.stream);
    let (padded_ptr, _padded_guard) = dispatch.num_tokens_post_padded.device_ptr(&ctx.stream);
    let (weights_ptr, _weights_guard) = dispatch.topk_weights.device_ptr(&ctx.stream);
    let workspace_len = workspace.len();
    let sorted_len = dispatch.sorted_token_ids.len();
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    let (c_tmp_ptr, _c_tmp_guard) = c_tmp.device_ptr_mut(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gemma4_marlin_nvfp4_moe_cuda(
            input_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            c_tmp_ptr as *mut f32,
            qweight_ptr as *const u8,
            scales_ptr as *const u8,
            global_ptr as *const f32,
            workspace_ptr as *mut i32,
            sorted_ptr as *const i32,
            expert_ptr as *const i32,
            padded_ptr as *const i32,
            weights_ptr as *const f32,
            i32::try_from(workspace_len)?,
            i32::try_from(sorted_len)?,
            i32::try_from(dispatch.block_size)?,
            i32::try_from(dispatch.top_k)?,
            dispatch.mul_topk_weights,
            i32::try_from(rows)?,
            i32::try_from(size_n)?,
            i32::try_from(size_k)?,
            0,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Rewrite a stacked projection's block scales into Marlin's order.
pub fn gemma4_marlin_nvfp4_prepare_scales(
    ctx: &DeviceContext,
    checkpoint: &CudaSlice<u8>,
    prepared: &mut CudaSlice<u8>,
    experts: usize,
    in_dim: usize,
    out_dim: usize,
    rescale: f32,
) -> Result<()> {
    let expected = experts * out_dim * (in_dim / 16);
    ensure!(
        in_dim.is_multiple_of(16) && checkpoint.len() >= expected && prepared.len() >= expected,
        "gemma4_marlin_nvfp4_prepare_scales: {experts} x {out_dim} x {in_dim} needs {expected} \
         bytes, have {} and {}",
        checkpoint.len(),
        prepared.len()
    );
    let (src_ptr, _src_guard) = checkpoint.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = prepared.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gemma4_marlin_nvfp4_prepare_scales_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            i32::try_from(experts)?,
            i32::try_from(in_dim)?,
            i32::try_from(out_dim)?,
            rescale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Rewrite a stacked four-bit projection into Marlin's B layout.
pub fn marlin_repack_4bit(
    ctx: &DeviceContext,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    experts: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<()> {
    let expected = experts * out_dim * in_dim / 2;
    ensure!(
        src.len() >= expected && dst.len() >= expected,
        "marlin_repack_4bit: {experts} x {out_dim} x {in_dim} needs {expected} bytes, have {} \
         and {}",
        src.len(),
        dst.len()
    );
    let (src_ptr, _src_guard) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::marlin_repack_4bit_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            i32::try_from(experts)?,
            i32::try_from(in_dim)?,
            i32::try_from(out_dim)?,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

/// Scratch the device-side dispatch builder writes through.
pub struct MoeAlignScratch<'a> {
    pub sorted_token_ids: &'a mut CudaSlice<i32>,
    pub expert_ids: &'a mut CudaSlice<i32>,
    pub num_tokens_post_padded: &'a mut CudaSlice<i32>,
    pub expert_offsets: &'a mut CudaSlice<u32>,
    pub expert_cursor: &'a mut CudaSlice<u32>,
}

/// Group the routed slots into the kernel's fixed-width blocks, on the device.
///
/// The capacities are the whole allocations, not this step's extent: the
/// builder clears them before it fills, so a step never reads what the last
/// one left behind.
pub fn marlin_moe_align_block_size(
    ctx: &DeviceContext,
    topk_idx: &CudaSlice<i32>,
    rows: usize,
    top_k: usize,
    experts: usize,
    block_size: usize,
    out: &mut MoeAlignScratch<'_>,
) -> Result<()> {
    let routes = rows
        .checked_mul(top_k)
        .ok_or_else(|| anyhow!("marlin_moe_align_block_size: {rows} x {top_k} overflows usize"))?;
    let max_padded = out.sorted_token_ids.len();
    let max_blocks = out.expert_ids.len();
    ensure!(
        topk_idx.len() >= routes
            && !out.num_tokens_post_padded.is_empty()
            && out.expert_offsets.len() > experts
            && out.expert_cursor.len() >= experts,
        "marlin_moe_align_block_size scratch too small for {routes} routes over {experts} experts"
    );
    ensure!(
        max_padded >= routes + experts * (block_size - 1),
        "marlin_moe_align_block_size: {max_padded} padded slots cannot hold {routes} routes with \
         one part filled block per expert"
    );
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (sorted_ptr, _sorted_guard) = out.sorted_token_ids.device_ptr_mut(&ctx.stream);
    let (expert_ptr, _expert_guard) = out.expert_ids.device_ptr_mut(&ctx.stream);
    let (padded_ptr, _padded_guard) = out.num_tokens_post_padded.device_ptr_mut(&ctx.stream);
    let (offsets_ptr, _offsets_guard) = out.expert_offsets.device_ptr_mut(&ctx.stream);
    let (cursor_ptr, _cursor_guard) = out.expert_cursor.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::marlin_moe_align_block_size_cuda(
            idx_ptr as *const i32,
            sorted_ptr as *mut i32,
            expert_ptr as *mut i32,
            padded_ptr as *mut i32,
            offsets_ptr as *mut u32,
            cursor_ptr as *mut u32,
            i32::try_from(rows)?,
            i32::try_from(top_k)?,
            0,
            i32::try_from(experts)?,
            i32::try_from(block_size)?,
            i32::try_from(max_padded)?,
            i32::try_from(max_blocks)?,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}
