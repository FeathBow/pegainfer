//! Shared device gate for the kernel GPU gate binaries.

use pegainfer_kernels::tensor::DeviceContext;

/// Never infer "no device" from an arbitrary error — a broken driver, a
/// failed stream or a context poisoned by an earlier `__trap()` all look
/// like one. The context attempt comes first because it also initialises
/// the driver; only its failure is diagnosed by `get_count()`, which is
/// itself ambiguous between "no driver" and "broken driver". A formal gate
/// must therefore set `PEGAINFER_REQUIRE_GPU=1` so skipping is impossible.
pub(crate) fn device_or_skip() -> Option<DeviceContext> {
    match DeviceContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            let count = cudarc::driver::result::device::get_count();
            match count {
                Err(_) | Ok(0) => {
                    assert!(
                        std::env::var("PEGAINFER_REQUIRE_GPU").as_deref() != Ok("1"),
                        "PEGAINFER_REQUIRE_GPU=1 but no usable CUDA device \
                         (context error: {e}; get_count: {count:?})"
                    );
                    eprintln!("skipping: no usable CUDA device (get_count: {count:?})");
                    None
                }
                Ok(n) => panic!(
                    "CUDA device present (get_count = {n}) but context creation \
                     failed: {e}. This is a broken environment or a poisoned \
                     context, not a missing device, and must not be skipped"
                ),
            }
        }
    }
}

/// Deterministic junk, distinct per element, so a row read from the wrong
/// page, layer or head cannot coincide with the right one. Not every binary
/// that includes this module fills a buffer.
#[allow(dead_code)]
pub(crate) fn fill(seed: u64, n: usize) -> Vec<half::bf16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            half::bf16::from_f32(((state >> 40) as f32 / 8_388_608.0) - 1.0)
        })
        .collect()
}
