//! The generated global-attention prefill against the one it stands in for,
//! on a pool and plan built the way the serving path builds them. Needs
//! neither weights nor a tokenizer, so it names which entry is wrong where
//! the serving gate only says the logits diverged.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.

#![cfg(feature = "gemma4")]

mod common;

use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_hd512_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::ops::gemma4_hd512_prefill_varlen_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::HiddenStates;

// Gemma 4's global family: 32 query heads over 4 key/value heads at head dim
// 512, paged at the attention key block.
const HD: usize = 512;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 4;
const PAGE_SIZE: usize = 64;
const NUM_LAYERS: usize = 10;
// The layer under test is not the first, so a wrong layer offset shows up.
const LAYER: usize = 5;
// Long enough to walk several key blocks and to leave the last page partial.
const SEQ_LEN: usize = 500;

#[test]
fn the_generated_global_prefill_matches_the_one_it_replaces() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    assert!(
        gemma4_hd512_prefill_is_built(),
        "this build carries the stub, so there is nothing to compare against"
    );

    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let pages = SEQ_LEN.div_ceil(PAGE_SIZE);
    // Page ids are scattered and the pool holds more pages than the request
    // uses, so a stride mistake reads somebody else's rows rather than its
    // own neighbour's.
    let pool_pages = pages * 3;
    let page_indices: Vec<i32> = (0..pages).map(|p| (p * 3 + 1) as i32).collect();
    let pool_elems = pool_pages * layout.page_stride;
    let pool_host = common::fill(0x51DE, pool_elems);
    let pool = ctx.stream.clone_htod(&pool_host).expect("pool upload");

    let q_host = common::fill(0xC0_FFEE, SEQ_LEN * NUM_Q_HEADS * HD);
    let q = HiddenStates {
        data: ctx.stream.clone_htod(&q_host).expect("q upload"),
        seq_len: SEQ_LEN,
        hidden_dim: NUM_Q_HEADS * HD,
    };

    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        &ctx,
        &page_indices,
        (SEQ_LEN - 1) % PAGE_SIZE + 1,
        0,
        SEQ_LEN,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        0,
    )
    .expect("prefill plan");

    let mut incumbent = HiddenStates::zeros(&ctx, NUM_Q_HEADS * HD, SEQ_LEN).expect("out a");
    let mut replacement = HiddenStates::zeros(&ctx, NUM_Q_HEADS * HD, SEQ_LEN).expect("out b");
    batch_prefill_paged_hd512_into(
        &ctx,
        &q,
        &pool,
        &layout,
        LAYER,
        &plan,
        &mut incumbent,
        NUM_Q_HEADS,
        1.0,
    )
    .expect("incumbent prefill");
    gemma4_hd512_prefill_varlen_into(
        &ctx,
        &q,
        &pool,
        &layout,
        LAYER,
        &plan,
        &mut replacement,
        NUM_Q_HEADS,
        1.0,
    )
    .expect("generated prefill");

    let a = incumbent.to_host(&ctx).expect("incumbent D2H");
    let b = replacement.to_host(&ctx).expect("replacement D2H");
    assert_eq!(a.len(), b.len());
    let (worst, worst_at) = common::worst_delta(&a, &b);
    let row = worst_at / (NUM_Q_HEADS * HD);
    let head = (worst_at % (NUM_Q_HEADS * HD)) / HD;
    eprintln!(
        "worst |delta| {worst} at row {row} head {head} lane {}; \
         incumbent {} replacement {}",
        worst_at % HD,
        a[worst_at],
        b[worst_at]
    );
    assert!(
        worst <= 0.05,
        "the two kernels disagree by {worst} at row {row} head {head}"
    );
}
