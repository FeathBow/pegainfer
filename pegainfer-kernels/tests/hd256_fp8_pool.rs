//! Device gates for Gemma 4's e4m3 paged-KV storage contract.

#![cfg(feature = "gemma4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_window_hd256_into;
use pegainfer_kernels::ops::paged_attention_batch_decode_hd256_into;
use pegainfer_kernels::ops::qkv_norm_rope_paged_prefill_hd256_plain_into;
use pegainfer_kernels::paged_kv::KvStorage;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 256;
const PAGE_SIZE: usize = 2;
const NUM_LAYERS: usize = 3;

fn packed_fp8(bytes: &[u8]) -> Vec<bf16> {
    bytes
        .chunks_exact(2)
        .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

fn raw_bytes(ctx: &DeviceContext, pool: &CudaSlice<bf16>) -> Vec<u8> {
    ctx.stream
        .clone_dtoh(pool)
        .expect("pool D2H")
        .into_iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn constant_states(ctx: &DeviceContext, value: f32, rows: usize) -> HiddenStates {
    HiddenStates::from_host(ctx, &vec![bf16::from_f32(value); HD * rows], HD, rows)
        .expect("states H2D")
}

fn identity_rope(ctx: &DeviceContext, rows: usize) -> (DeviceVec, DeviceVec) {
    let cos = vec![bf16::ONE; HD * rows];
    let sin = vec![bf16::ZERO; HD * rows];
    (
        DeviceVec::from_host(ctx, &cos).expect("cos H2D"),
        DeviceVec::from_host(ctx, &sin).expect("sin H2D"),
    )
}

#[test]
fn fp8_prep_stores_exact_bytes_at_layout_offsets() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let layout = PagedKvLayout::with_storage(NUM_LAYERS, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride * 3 / 2)
        .expect("pool alloc");
    let q = constant_states(&ctx, 1.0, 3);
    let k = constant_states(&ctx, 1.0, 3);
    let v = constant_states(&ctx, 0.5, 3);
    let mut q_out = HiddenStates::zeros(&ctx, HD, 3).expect("q_out alloc");
    let weights = DeviceVec::from_host(&ctx, &vec![bf16::from_f32(2.0); HD]).expect("weights H2D");
    let (cos, sin) = identity_rope(&ctx, 3);
    let pages = ctx.stream.clone_htod(&[2i32, 0]).expect("pages H2D");
    qkv_norm_rope_paged_prefill_hd256_plain_into(
        &ctx, &q, &k, &v, &mut q_out, 0, &pool, &layout, &weights, &weights, &cos, &sin, 1, &pages,
        0, 0, 0, 3, 1, 1, HD, 0.0,
    )
    .expect("fp8 prep");
    let got = raw_bytes(&ctx, &pool);
    let mut expected = vec![0u8; layout.page_stride * 3];
    for (token, page) in [2usize, 2, 0].into_iter().enumerate() {
        let slot = token % PAGE_SIZE;
        let layer = page * layout.page_stride + layout.layer_stride;
        let k = layer + slot * HD;
        let v = layer + layout.kv_block_len + slot * HD;
        expected[k..k + HD].fill(0x40);
        expected[v..v + HD].fill(0x38);
    }
    assert_eq!(got, expected);
}

fn semantic_pool(ctx: &DeviceContext, storage: KvStorage, pages: usize) -> CudaSlice<bf16> {
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, storage);
    let values = [1.0f32, 2.0, 0.5, -1.0];
    match storage {
        KvStorage::Bf16 => {
            let mut host = vec![bf16::ZERO; layout.page_stride * pages];
            for page in 0..pages {
                for slot in 0..PAGE_SIZE {
                    let base = page * layout.page_stride + slot * HD;
                    host[base..base + HD].fill(bf16::from_f32(values[slot]));
                    let v = base + layout.kv_block_len;
                    host[v..v + HD].fill(bf16::from_f32(values[slot + 2]));
                }
            }
            ctx.stream.clone_htod(&host).expect("bf16 pool H2D")
        }
        KvStorage::E4m3 => {
            let mut bytes = vec![0u8; layout.page_stride * pages];
            for page in 0..pages {
                for slot in 0..PAGE_SIZE {
                    let base = page * layout.page_stride + slot * HD;
                    bytes[base..base + HD].fill([0x38, 0x40][slot]);
                    let v = base + layout.kv_block_len;
                    bytes[v..v + HD].fill([0x30, 0xb8][slot]);
                }
            }
            ctx.stream
                .clone_htod(&packed_fp8(&bytes))
                .expect("fp8 pool H2D")
        }
    }
}

fn attend(ctx: &DeviceContext, storage: KvStorage) -> Vec<u16> {
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, storage);
    let pool = semantic_pool(ctx, storage, 1);
    let plan =
        PrefillPagedPlan::new_with_cta_tile_q(ctx, &[0], 2, 1, 1, 1, 1, HD, 0).expect("plan");
    let q = constant_states(ctx, 0.0, 1);
    let mut output = HiddenStates::zeros(ctx, HD, 1).expect("output alloc");
    batch_prefill_paged_window_hd256_into(
        ctx,
        &q,
        &pool,
        &layout,
        0,
        &plan,
        &mut output,
        1,
        1.0,
        -1,
    )
    .expect("attention");
    output
        .to_host(ctx)
        .expect("output D2H")
        .into_iter()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect()
}

#[test]
fn fp8_window_read_matches_bf16_for_exact_values() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    assert_eq!(attend(&ctx, KvStorage::E4m3), attend(&ctx, KvStorage::Bf16));
}

fn geometry_probe(ctx: &DeviceContext, prefix_rows: usize) -> Vec<u16> {
    let pages = prefix_rows.div_ceil(PAGE_SIZE) + 1;
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let pool = semantic_pool(ctx, KvStorage::E4m3, pages);
    let (page_lists, starts, lengths, lasts) = if prefix_rows == 0 {
        (vec![vec![0]], vec![1], vec![1], vec![2])
    } else {
        let prefix_pages: Vec<i32> = (0..pages as i32 - 1).collect();
        (
            vec![prefix_pages, vec![pages as i32 - 1]],
            vec![0, 1],
            vec![prefix_rows, 1],
            vec![(prefix_rows - 1) % PAGE_SIZE + 1, 2],
        )
    };
    let plan = PrefillPagedPlan::new_batch_with_cta_tile_q(
        ctx,
        &page_lists,
        &lasts,
        &starts,
        &lengths,
        1,
        1,
        HD,
        0,
    )
    .expect("batch plan");
    let q = constant_states(ctx, 0.0, prefix_rows + 1);
    let mut output = HiddenStates::zeros(ctx, HD, prefix_rows + 1).expect("output alloc");
    batch_prefill_paged_window_hd256_into(
        ctx,
        &q,
        &pool,
        &layout,
        0,
        &plan,
        &mut output,
        1,
        1.0,
        -1,
    )
    .expect("attention");
    let host = output.to_host(ctx).expect("output D2H");
    host[prefix_rows * HD..]
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect()
}

#[test]
fn fp8_window_read_is_geometry_invariant_for_the_probed_row() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let lone = geometry_probe(&ctx, 0);
    let packed = geometry_probe(&ctx, 300);
    if let Some(index) = lone.iter().zip(&packed).position(|(a, b)| a != b) {
        panic!(
            "geometry dependence at output[{index}]: lone={:#06x}, packed={:#06x}",
            lone[index], packed[index]
        );
    }
}

#[test]
fn decode_wrapper_without_fp8_twin_refuses_e4m3() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let layout = PagedKvLayout::with_storage(1, 1, HD, PAGE_SIZE, KvStorage::E4m3);
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride / 2)
        .expect("pool");
    let state = constant_states(&ctx, 0.0, 1);
    let mut output = HiddenStates::zeros(&ctx, HD, 1).expect("output");
    let meta = ctx.stream.clone_htod(&[0i32]).expect("metadata");
    let indptr = ctx.stream.clone_htod(&[0i32, 1]).expect("indptr");
    let err = paged_attention_batch_decode_hd256_into(
        &ctx,
        &state,
        &state,
        &state,
        &pool,
        &layout,
        0,
        &meta,
        &indptr,
        &meta,
        &meta,
        &meta,
        &meta,
        &meta,
        &mut output,
        1,
        1,
    )
    .expect_err("unsupported fp8 wrapper must reject");
    assert!(err.to_string().contains("has no fp8 KV path"), "{err}");
}
