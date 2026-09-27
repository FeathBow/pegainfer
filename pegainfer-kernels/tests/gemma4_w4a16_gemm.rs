//! The W4A16 path against its definition, `y = x @ ((q - 8) * s)^T`, on every
//! 31B linear shape: the load-time rewrite and dequantization bit-exact, the
//! TileLang GEMMs at each row bucket (and at a count that pads into one, and
//! one past the largest, which dequantizes) against a host product, a row's
//! bits the same in every bucket, and a relaunch reproducing its output.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure. A build without the generated GEMMs skips with a message.

#![cfg(feature = "gemma4")]

mod common;

use half::bf16;
use pegainfer_kernels::ops::W4a16Matrix;
use pegainfer_kernels::ops::W4a16Scratch;
use pegainfer_kernels::ops::gemma4_w4a16_gemm_into;
use pegainfer_kernels::ops::gemma4_w4a16_geometry;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::HiddenStates;

const GROUP: usize = 32;
const SHAPES: [(usize, usize); 6] = [
    (16384, 5376),
    (18432, 5376),
    (5376, 8192),
    (5376, 16384),
    (43008, 5376),
    (5376, 21504),
];
/// Every `COL_STRIDE`-th output column is checked against the host product.
const COL_STRIDE: usize = 37;

struct Checkpoint {
    packed: Vec<u32>,
    scales: Vec<bf16>,
}

impl Checkpoint {
    fn random(seed: u64, rows: usize, cols: usize) -> Self {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let packed = (0..rows * cols / 8).map(|_| next() as u32).collect();
        let scales = (0..rows * cols / GROUP)
            .map(|_| bf16::from_f32(0.001 + (next() >> 40) as f32 / 16_777_216.0 * 0.02))
            .collect();
        Self { packed, scales }
    }

    /// `(q - 8) * s` rounded to bf16, the reference's weight.
    fn weight(&self, rows: usize, cols: usize) -> Vec<bf16> {
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let q = (self.packed[(r * cols + c) / 8] >> (4 * (c % 8))) & 0xF;
                let s = self.scales[r * (cols / GROUP) + c / GROUP].to_f32();
                out.push(bf16::from_f32((q as f32 - 8.0) * s));
            }
        }
        out
    }

    fn upload(&self, ctx: &DeviceContext, rows: usize, cols: usize) -> W4a16Matrix {
        let packed: Vec<u8> = self.packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales: Vec<u8> = self
            .scales
            .iter()
            .flat_map(|s| s.to_bits().to_le_bytes())
            .collect();
        let packed = ctx.stream.clone_htod(&packed).expect("packed upload");
        let scales = ctx.stream.clone_htod(&scales).expect("scales upload");
        W4a16Matrix::from_checkpoint(ctx, &packed, &scales, rows, cols).expect("rewrite")
    }
}

fn host(ctx: &DeviceContext, states: &HiddenStates, rows: usize) -> Vec<f32> {
    let all = ctx.stream.clone_dtoh(&states.data).expect("download");
    all[..rows * states.hidden_dim]
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

/// `rows` rows of `x` through the GEMM into a 16-row-capacity output.
fn run(
    ctx: &DeviceContext,
    weight: &W4a16Matrix,
    x: &[bf16],
    rows: usize,
    scratch: &mut W4a16Scratch,
) -> Vec<f32> {
    let capacity = rows.max(16);
    let mut padded = x[..rows * weight.cols].to_vec();
    padded.resize(capacity * weight.cols, bf16::from_f32(7.0));
    let input = HiddenStates {
        data: ctx.stream.clone_htod(&padded).expect("x upload"),
        hidden_dim: weight.cols,
        seq_len: rows,
    };
    let mut out = HiddenStates::zeros(ctx, weight.rows, capacity).expect("out");
    out.seq_len = rows;
    gemma4_w4a16_gemm_into(ctx, weight, &input, scratch, &mut out).expect("gemm");
    host(ctx, &out, rows)
}

#[test]
fn w4a16_gemm_matches_its_definition_on_every_31b_shape() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    if gemma4_w4a16_geometry().is_none() {
        eprintln!("skipping: this build carries no W4A16 GEMMs");
        return;
    }
    let largest = SHAPES.iter().map(|(n, k)| n * k).max().unwrap();
    let mut scratch = W4a16Scratch::new(&ctx, largest).expect("scratch");
    for (shape, &(n, k)) in SHAPES.iter().enumerate() {
        let checkpoint = Checkpoint::random(0x51A7 + shape as u64, n, k);
        let weight = checkpoint.upload(&ctx, n, k);
        let reference = checkpoint.weight(n, k);

        let mut dense = DeviceMatrix {
            data: ctx.stream.alloc_zeros(n * k).expect("dense"),
            rows: 0,
            cols: 0,
        };
        weight.dequant_into(&ctx, &mut dense).expect("dequant");
        let got = ctx.stream.clone_dtoh(&dense.data).expect("dense download");
        let mismatch = got
            .iter()
            .zip(&reference)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            mismatch.is_none(),
            "{n} x {k}: dequantized weight differs at {mismatch:?}"
        );

        let x = common::fill(0xF00D + shape as u64, 20 * k);
        let mut first_rows: Vec<Vec<f32>> = Vec::new();
        for rows in [1, 2, 3, 4, 8, 16, 20] {
            let y = run(&ctx, &weight, &x, rows, &mut scratch);
            for r in 0..rows {
                let mut worst = 0.0f32;
                let mut peak = 0.0f32;
                for c in (0..n).step_by(COL_STRIDE) {
                    let want: f32 = (0..k)
                        .map(|i| x[r * k + i].to_f32() * reference[c * k + i].to_f32())
                        .sum();
                    worst = worst.max((y[r * n + c] - want).abs());
                    peak = peak.max(want.abs());
                }
                assert!(
                    worst <= 2e-2 * peak.max(1.0),
                    "{n} x {k} at {rows} rows, row {r}: worst {worst} against peak {peak}"
                );
            }
            if rows <= 16 {
                first_rows.push(y[..n].to_vec());
            }
            if rows == 16 {
                let again = run(&ctx, &weight, &x, rows, &mut scratch);
                assert_eq!(y, again, "{n} x {k}: a relaunch changed the output");
            }
        }
        for (i, row) in first_rows.iter().enumerate().skip(1) {
            assert!(
                row.iter()
                    .zip(&first_rows[0])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{n} x {k}: row 0 differs between bucket runs 0 and {i}"
            );
        }
    }
}
