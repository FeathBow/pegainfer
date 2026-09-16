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
use crate::ops::PrefillPagedPlan;
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
    let row_elems = layout.num_kv_heads * head_dim;
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
    anyhow::ensure!(
        layer < layout.num_layers,
        "gemma4 hd512 prefill layer {layer} >= layout.num_layers {}",
        layout.num_layers
    );
    q.checked_extent("gemma4 hd512 prefill q")?;
    output.checked_extent("gemma4 hd512 prefill output")?;
    // The pool is addressed in rows of [num_kv_heads, head_dim]; one page
    // holds every layer's K then V, so a layer's K block starts `layer_row`
    // into the page and its V block one page further.
    anyhow::ensure!(
        layout.page_stride.is_multiple_of(row_elems) && kv_buffer.len().is_multiple_of(row_elems),
        "gemma4 hd512 prefill pool does not divide into rows of {row_elems}"
    );
    let rows_per_page = crate::ops::checked_i32(
        layout.page_stride / row_elems,
        "gemma4 hd512 prefill rows_per_page",
    )?;
    let layer_row = crate::ops::checked_i32(
        layer * 2 * layout.page_size,
        "gemma4 hd512 prefill layer_row",
    )?;
    let pool_rows = crate::ops::checked_i32(
        kv_buffer.len() / row_elems,
        "gemma4 hd512 prefill pool_rows",
    )?;
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
