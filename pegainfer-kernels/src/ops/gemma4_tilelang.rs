//! Gemma 4's TileLang global-attention prefill, behind the same interface as
//! the FlashInfer one it stands in for. The prefill path folds
//! `1/sqrt(head_dim)` into the query rows upstream and hands the kernel a
//! scale of one; a kernel with the factor baked in would apply it twice.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ops::Hd512DecodeMetadata;
use crate::ops::PrefillPagedPlan;
use crate::ops::attention::checked_row_offset;
use crate::paged_kv::PagedKvLayout;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;

/// `(num_qo_heads, num_kv_heads, head_dim, page_size)` the generated bodies
/// were compiled for, `None` in a stub build. The launcher refuses any other.
pub fn gemma4_hd512_prefill_geometry() -> Option<(usize, usize, usize, usize)> {
    let stated = option_env!("PEGAINFER_GEMMA4_TILELANG_GEOMETRY")?;
    let mut parts = stated.split(',').map(str::trim).map(str::parse::<usize>);
    let mut next = || parts.next()?.ok();
    Some((next()?, next()?, next()?, next()?))
}

/// Dynamic shared memory one block of the generated bodies opts into, `None`
/// in a stub build. A device whose per-block limit is under it cannot run them.
pub fn gemma4_hd512_prefill_smem() -> Option<usize> {
    option_env!("PEGAINFER_GEMMA4_TILELANG_SMEM")?.parse().ok()
}

/// The arch the generated bodies were assembled for, `None` in a stub build.
/// One arch per generation, so any other device has no image to run.
pub fn gemma4_hd512_prefill_arch() -> Option<&'static str> {
    option_env!("PEGAINFER_GEMMA4_TILELANG_ARCH").filter(|arch| !arch.is_empty())
}

/// Whether this build carries the generated bodies or the refusing stub. The
/// cfg is this crate's, so a model crate cannot read it directly; the stub
/// answers `cudaErrorNotSupported`, but only after the weights are loaded.
pub fn gemma4_hd512_prefill_is_built() -> bool {
    cfg!(gemma4_tilelang)
}

/// Causal GQA prefill over the global family's paged pool, ragged across the
/// step's prompt segments. Same arguments as `batch_prefill_paged_hd512_into`;
/// of the plan it reads the page table and the query boundaries, the latter on
/// the device for the walk and on the host for the launcher's grid.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_hd512_prefill_varlen_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    plan: &PrefillPagedPlan,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    sm_scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        sm_scale.is_finite(),
        "gemma4 hd512 prefill sm_scale {sm_scale} must be finite"
    );
    let head_dim = layout.head_dim;
    anyhow::ensure!(
        q.hidden_dim == num_qo_heads * head_dim,
        "gemma4 hd512 prefill q.hidden_dim {} != num_qo_heads {num_qo_heads} * {head_dim}",
        q.hidden_dim,
    );
    anyhow::ensure!(
        output.hidden_dim == q.hidden_dim,
        "gemma4 hd512 prefill output.hidden_dim {} != q.hidden_dim {}",
        output.hidden_dim,
        q.hidden_dim,
    );
    // The plan's last boundary is how many rows it packed; asking it that way
    // keeps one source rather than a second recorded total.
    let host_q_indptr = plan.q_indptr_host();
    let packed_rows = *host_q_indptr.last().unwrap_or(&0) as usize;
    anyhow::ensure!(
        q.seq_len >= packed_rows && output.seq_len >= packed_rows,
        "gemma4 hd512 prefill rows (q {}, out {}) below the plan's {packed_rows}",
        q.seq_len,
        output.seq_len,
    );
    q.checked_extent("gemma4 hd512 prefill q")?;
    output.checked_extent("gemma4 hd512 prefill output")?;
    // The kernel addresses the pool as rows of [num_kv_heads, row width]; the
    // layout says where a layer's block starts in that view and how many rows
    // the pool holds. The row's format goes as the count the bodies tell it
    // by: zero for the split rows, whose V is one page of rows after K, and
    // the rotated columns a folded row keeps otherwise. The launcher refuses
    // a count its bodies were not lowered for.
    let rows = layout.row_geometry(kv_buffer.len(), layer, "gemma4 hd512 prefill")?;
    let fold_rotary = crate::ops::checked_i32(
        layout.format.fold_rotary(),
        "gemma4 hd512 prefill fold_rotary",
    )?;
    let rows_per_page =
        crate::ops::checked_i32(rows.rows_per_page, "gemma4 hd512 prefill rows_per_page")?;
    let layer_row = crate::ops::checked_i32(rows.layer_row, "gemma4 hd512 prefill layer_row")?;
    let pool_rows = crate::ops::checked_i32(rows.pool_rows, "gemma4 hd512 prefill pool_rows")?;
    let q_rows = crate::ops::checked_i32(q.seq_len, "gemma4 hd512 prefill q_rows")?;
    let page_size = crate::ops::checked_i32(layout.page_size, "gemma4 hd512 prefill page_size")?;

    let num_qo_heads_i32 =
        crate::ops::checked_i32(num_qo_heads, "gemma4 hd512 prefill num_qo_heads")?;
    let num_kv_heads_i32 =
        crate::ops::checked_i32(layout.num_kv_heads, "gemma4 hd512 prefill num_kv_heads")?;
    let batch = plan.batch_size();

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (kv_ptr, _gkv) = kv_buffer.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = plan.page_indices_d().device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = plan.page_indptr_d().device_ptr(&ctx.stream);
    let (qi_ptr, _gqi) = plan.q_indptr_d().device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = plan.last_page_len_d().device_ptr(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    let rc = unsafe {
        ffi::gemma4_hd512_prefill_varlen(
            q_ptr as *const core::ffi::c_void,
            kv_ptr as *const core::ffi::c_void,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            qi_ptr as *const i32,
            host_q_indptr.as_ptr(),
            lpl_ptr as *const i32,
            out_ptr as *mut core::ffi::c_void,
            batch,
            q_rows,
            pool_rows,
            rows_per_page,
            layer_row,
            page_size,
            fold_rotary,
            num_qo_heads_i32,
            num_kv_heads_i32,
            sm_scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    checked_launch(rc)
}

/// The fn type both global decode reads share, so a caller holds one and the
/// choice between them is a flag rather than a shape the call sites know.
pub type GlobalDecodeAttend = fn(
    &DeviceContext,
    &HiddenStates,
    usize,
    &CudaSlice<bf16>,
    &PagedKvLayout,
    usize,
    &Hd512DecodeMetadata,
    &CudaSlice<i32>,
    &CudaSlice<u8>,
    &mut CudaSlice<bf16>,
    &mut CudaSlice<f32>,
    usize,
    &mut HiddenStates,
    usize,
    f32,
) -> Result<()>;

/// Split-KV decode over the global family's paged pool, for the step's
/// decode rows.
///
/// Same arguments as `paged_attention_batch_decode_split_kv_hd512_into`, and
/// the same meaning for each. The plan, the workspace and the per-slot state
/// are the serving path's own; what differs is the kernel that walks them
/// and merges them, and the partial state's meaning between the two passes,
/// which is this kernel's to define since it writes and reads both ends.
#[allow(clippy::too_many_arguments)]
pub fn gemma4_hd512_decode_split_kv_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    row_offset: usize,
    kv_buffer: &CudaSlice<bf16>,
    layout: &PagedKvLayout,
    layer: usize,
    meta: &Hd512DecodeMetadata,
    split_o_indptr_d: &CudaSlice<i32>,
    split_valid_mask_d: &CudaSlice<u8>,
    split_tmp_v: &mut CudaSlice<bf16>,
    split_tmp_s: &mut CudaSlice<f32>,
    split_padded_slots: usize,
    output: &mut HiddenStates,
    num_qo_heads: usize,
    sm_scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        sm_scale.is_finite(),
        "gemma4 hd512 decode sm_scale {sm_scale} must be finite"
    );
    let head_dim = layout.head_dim;
    anyhow::ensure!(
        row_offset < q.seq_len,
        "gemma4 hd512 decode row_offset {row_offset} leaves no rows of {}",
        q.seq_len
    );
    let batch = q.seq_len - row_offset;
    meta.validate(batch)?;
    anyhow::ensure!(
        output.seq_len == q.seq_len,
        "gemma4 hd512 decode output.seq_len {} != q.seq_len {}",
        output.seq_len,
        q.seq_len
    );
    anyhow::ensure!(
        q.hidden_dim == num_qo_heads * head_dim && output.hidden_dim == q.hidden_dim,
        "gemma4 hd512 decode q/out hidden_dim ({}, {}) != num_qo_heads {num_qo_heads} * {head_dim}",
        q.hidden_dim,
        output.hidden_dim,
    );
    anyhow::ensure!(
        split_padded_slots >= batch,
        "gemma4 hd512 decode padded_slots {split_padded_slots} < batch {batch}"
    );
    anyhow::ensure!(
        meta.request_indices().len() >= split_padded_slots
            && meta.kv_tile_indices().len() >= split_padded_slots
            && split_valid_mask_d.len() >= split_padded_slots,
        "gemma4 hd512 decode plan arrays shorter than padded_slots {split_padded_slots}"
    );
    anyhow::ensure!(
        split_o_indptr_d.len() > batch,
        "gemma4 hd512 decode o_indptr len {} < batch {batch} + 1",
        split_o_indptr_d.len()
    );
    anyhow::ensure!(
        split_tmp_v.len() >= split_padded_slots * q.hidden_dim
            && split_tmp_s.len() >= split_padded_slots * num_qo_heads,
        "gemma4 hd512 decode workspace shorter than padded_slots {split_padded_slots}"
    );
    // The tile indices count chunks; the body walks pages, so a chunk has
    // to be whole pages or the two would name different keys.
    anyhow::ensure!(
        meta.chunk_tokens().is_multiple_of(layout.page_size),
        "gemma4 hd512 decode chunk {} is not whole pages of {}",
        meta.chunk_tokens(),
        layout.page_size
    );
    let rows = layout.row_geometry(kv_buffer.len(), layer, "gemma4 hd512 decode")?;
    let fold_rotary = crate::ops::checked_i32(
        layout.format.fold_rotary(),
        "gemma4 hd512 decode fold_rotary",
    )?;
    let rows_per_page =
        crate::ops::checked_i32(rows.rows_per_page, "gemma4 hd512 decode rows_per_page")?;
    let layer_row = crate::ops::checked_i32(rows.layer_row, "gemma4 hd512 decode layer_row")?;
    let pool_rows = crate::ops::checked_i32(rows.pool_rows, "gemma4 hd512 decode pool_rows")?;
    let q_rows = crate::ops::checked_i32(q.seq_len, "gemma4 hd512 decode q_rows")?;
    let batch_i32 = crate::ops::checked_i32(batch, "gemma4 hd512 decode batch")?;
    let padded_i32 =
        crate::ops::checked_i32(split_padded_slots, "gemma4 hd512 decode padded slots")?;
    let row_offset_i32 = crate::ops::checked_i32(row_offset, "gemma4 hd512 decode row_offset")?;
    let page_size = crate::ops::checked_i32(layout.page_size, "gemma4 hd512 decode page_size")?;
    let chunk_tokens = crate::ops::checked_i32(meta.chunk_tokens(), "gemma4 hd512 decode chunk")?;
    let num_qo_heads_i32 =
        crate::ops::checked_i32(num_qo_heads, "gemma4 hd512 decode num_qo_heads")?;
    let num_kv_heads_i32 =
        crate::ops::checked_i32(layout.num_kv_heads, "gemma4 hd512 decode num_kv_heads")?;
    // The rows are addressed from the buffer's start on the device: the
    // launcher takes the offset rather than an advanced pointer, so the
    // descriptor it builds over the query rows sees the whole buffer.
    checked_row_offset(q, row_offset, batch, "gemma4 hd512 decode q")?;
    checked_row_offset(output, row_offset, batch, "gemma4 hd512 decode output")?;

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (kv_ptr, _gkv) = kv_buffer.device_ptr(&ctx.stream);
    let (pi_ptr, _gpi) = meta.page_indices().device_ptr(&ctx.stream);
    let (pip_ptr, _gpip) = meta.page_indptr().device_ptr(&ctx.stream);
    let (lpl_ptr, _glpl) = meta.last_page_len().device_ptr(&ctx.stream);
    let (ri_ptr, _gri) = meta.request_indices().device_ptr(&ctx.stream);
    let (kti_ptr, _gkti) = meta.kv_tile_indices().device_ptr(&ctx.stream);
    let (svm_ptr, _gsvm) = split_valid_mask_d.device_ptr(&ctx.stream);
    let (soi_ptr, _gsoi) = split_o_indptr_d.device_ptr(&ctx.stream);
    let (stv_ptr, _gstv) = split_tmp_v.device_ptr_mut(&ctx.stream);
    let (sts_ptr, _gsts) = split_tmp_s.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    let rc = unsafe {
        ffi::gemma4_hd512_decode_split_kv(
            q_ptr as *const core::ffi::c_void,
            kv_ptr as *const core::ffi::c_void,
            pi_ptr as *const i32,
            pip_ptr as *const i32,
            lpl_ptr as *const i32,
            ri_ptr as *const i32,
            kti_ptr as *const i32,
            svm_ptr as *const u8,
            soi_ptr as *const i32,
            stv_ptr as *mut core::ffi::c_void,
            sts_ptr as *mut f32,
            out_ptr as *mut core::ffi::c_void,
            batch_i32,
            padded_i32,
            row_offset_i32,
            q_rows,
            pool_rows,
            rows_per_page,
            layer_row,
            page_size,
            fold_rotary,
            chunk_tokens,
            num_qo_heads_i32,
            num_kv_heads_i32,
            sm_scale,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    checked_launch(rc)
}

/// `cudaErrorNotSupported` is the stub tier saying this build has no kernel;
/// anything else is a call the bodies refused or a launch that failed. Naming
/// the first is the difference between a build question and a bug hunt.
fn checked_launch(rc: i32) -> Result<()> {
    const NOT_SUPPORTED: i32 = 801;
    match rc {
        0 => Ok(()),
        NOT_SUPPORTED => anyhow::bail!(
            "gemma4 hd512 prefill is not in this build: it fell back to the \
             stub tier, so no TileLang and no pre-generated directory were \
             available when pegainfer-kernels was compiled"
        ),
        other => anyhow::bail!(
            "gemma4 hd512 prefill failed: cudaError={other} (1 = the call is \
             outside the extents, batch or page size the bodies were built for)"
        ),
    }
}
