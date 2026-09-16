//! Gemma 4 TileLang-generated global-attention prefill (AOT), built by the
//! `tilelang` section of `build.rs` from `pegainfer-gemma4/kernels/generate.py`.
//!
//! The symbol is a hand-written launcher returning `cudaError_t` as `int`:
//! `cudaErrorInvalidValue` outside what the body was built for,
//! `cudaErrorNotSupported` from the stub tier. It owns its own grid, which is
//! why the query boundaries arrive twice, on the device and on the host.

use core::ffi::c_void;

use cudarc::driver::sys::CUstream;

unsafe extern "C" {
    /// Causal GQA prefill for one layer's global family, ragged across the
    /// step's prompt segments. `q` and `out` are bf16 rows of
    /// `[num_qo_heads, head_dim]`, `kv` the pool in rows of
    /// `[num_kv_heads, head_dim]`, and the indptrs the prefill plan's own,
    /// `host_q_indptr` its host copy. `sm_scale` multiplies the scores; the
    /// serving path folds `1/sqrt(head_dim)` upstream and passes one. The
    /// extents, `batch`, `page_size` and the head counts are refused rather
    /// than truncated when they exceed what the bodies were built for.
    pub fn gemma4_hd512_prefill_varlen(
        q: *const c_void,
        kv: *const c_void,
        page_indices: *const i32,
        page_indptr: *const i32,
        q_indptr: *const i32,
        host_q_indptr: *const i32,
        last_page_len: *const i32,
        out: *mut c_void,
        batch: i32,
        q_rows: i32,
        pool_rows: i32,
        rows_per_page: i32,
        layer_row: i32,
        page_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;
}
