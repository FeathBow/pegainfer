//! Times both hd512 global-attention prefills at the shapes the chunked walk
//! launches. One kernel per process, so `ncu` needs no filter; the two arms
//! share one pool and one plan and alternate which leads each round.
//!
//!   cargo run --release --example hd512_prefill_bench -- <kv_len>[,<kv_len>...]
//!
//! Each kv_len is one chunk: the query block is CHUNK rows ending at kv_len,
//! which is what lines the causal mask up with the engine's.
use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_hd512_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::ops::gemma4_hd512_prefill_varlen_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
// The global family's page size. The generated kernel refuses anything else,
// and the incumbent at page 16 was measured to be worth 1.02-1.04x, so a
// comparison at 16 would be measuring a shape neither side ships.
const PAGE: usize = 64;
const CHUNK: usize = 8192;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 4;
const LAYERS: usize = 10;

/// The seam the serving path switches on, so the bench cannot drift from it.
type Attend = fn(
    &DeviceContext,
    &HiddenStates,
    &CudaSlice<bf16>,
    &PagedKvLayout,
    usize,
    &PrefillPagedPlan,
    &mut HiddenStates,
    usize,
    f32,
) -> Result<()>;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n.is_multiple_of(2) {
        f64::midpoint(xs[n / 2 - 1], xs[n / 2])
    } else {
        xs[n / 2]
    }
}

fn main() -> Result<()> {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "8192".into());
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let rounds: usize = std::env::var("BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let ctx = DeviceContext::new()?;

    let mut arms: Vec<(&str, Attend)> = vec![("incumbent", batch_prefill_paged_hd512_into)];
    if gemma4_hd512_prefill_is_built() {
        arms.push(("tilelang", gemma4_hd512_prefill_varlen_into));
    } else {
        println!("note: this build carries the stub, so only the incumbent runs");
    }

    println!(
        "{:>9} {:>7} {:>10} {:>10} {:>11} {:>11} {:>7} {:>12}",
        "kv_len", "pages", "incumb ms", "tilelang", "incumb TF", "tilelang TF", "ratio", "req GiB"
    );
    for token in arg.split(',') {
        let kv_len: usize = token.trim().parse()?;
        anyhow::ensure!(
            kv_len >= CHUNK && kv_len.is_multiple_of(PAGE),
            "kv_len {kv_len} must be >= {CHUNK} and a multiple of {PAGE}"
        );
        let pages = kv_len.div_ceil(PAGE);
        let layout = PagedKvLayout::new(LAYERS, KV_HEADS, HD, PAGE);
        // One page holds every layer's K and V for PAGE tokens, which is what
        // puts a single layer's pages apart in the real pool.
        let pool_elems = pages * LAYERS * 2 * PAGE * KV_HEADS * HD;
        let pool = ctx.stream.alloc_zeros::<bf16>(pool_elems)?;
        let page_indices: Vec<i32> = (0..pages as i32).collect();
        let plan = PrefillPagedPlan::new_with_cta_tile_q(
            &ctx,
            &page_indices,
            (kv_len - 1) % PAGE + 1,
            kv_len - CHUNK,
            CHUNK,
            Q_HEADS,
            KV_HEADS,
            HD,
            0,
        )?;
        let q = HiddenStates::zeros(&ctx, Q_HEADS * HD, CHUNK)?;
        let mut out = HiddenStates::zeros(&ctx, Q_HEADS * HD, CHUNK)?;
        let scale = (HD as f32).powf(-0.5);
        let time = |attend: Attend, out: &mut HiddenStates| -> Result<f64> {
            attend(
                &ctx,
                &q,
                &pool,
                &layout,
                LAYERS / 2,
                &plan,
                out,
                Q_HEADS,
                scale,
            )?;
            ctx.stream.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                attend(
                    &ctx,
                    &q,
                    &pool,
                    &layout,
                    LAYERS / 2,
                    &plan,
                    out,
                    Q_HEADS,
                    scale,
                )?;
            }
            ctx.stream.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
        };

        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            // Alternate which arm leads, so whichever penalty falls on going
            // first is not always paid by the same one.
            let order: Vec<usize> = if round.is_multiple_of(2) {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for i in order {
                samples[i].push(time(arms[i].1, &mut out)?);
            }
        }

        // Causal, bottom-right aligned: the mean key count per query row is
        // kv_len - CHUNK/2, and each of the two matmuls costs two flops per MAC.
        let flops =
            4.0 * CHUNK as f64 * (kv_len as f64 - CHUNK as f64 / 2.0) * Q_HEADS as f64 * HD as f64;
        // Every CTA rescans its kv head's history, so requested bytes scale with
        // the CTA count, which is the whole point of the comparison.
        let ctas = (CHUNK * Q_HEADS) as f64 / 32.0;
        let req = ctas * (kv_len as f64 - CHUNK as f64 / 2.0) * (HD * 2 * 2) as f64;
        let ms: Vec<f64> = samples.into_iter().map(median).collect();
        let tf = |m: f64| flops / (m / 1e3) / 1e12;
        let (b_ms, b_tf, ratio) = match ms.get(1) {
            Some(&m) => (format!("{m:.3}"), format!("{:.1}", tf(m)), ms[0] / m),
            None => ("-".into(), "-".into(), f64::NAN),
        };
        println!(
            "{kv_len:>9} {pages:>7} {:>10.3} {b_ms:>10} {:>11.1} {b_tf:>11} {ratio:>7.3} {:>12.1}",
            ms[0],
            tf(ms[0]),
            req / (1u64 << 30) as f64
        );
    }
    Ok(())
}
