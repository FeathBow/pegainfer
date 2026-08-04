//! Per-request KV state across the two families, and the admission that
//! keeps their pools consistent.

use anyhow::Result;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::kv_pool::KvState;

/// Cache slots and absolute positions coincide while `resident_origin`
/// stays 0.
pub(crate) struct GemmaKv {
    pub(crate) local: KvState,
    pub(crate) global: KvState,
    /// Zero while nothing has been evicted; moving it is what splits cache
    /// slots from absolute positions.
    pub(crate) local_resident_origin: usize,
}

/// A refused side drops the other reservation, leaving both pools at their
/// pre-request occupancy. The page count is the exact frontier account: a
/// ceiling over the post-step kv_len, per family.
pub(crate) fn admit_tokens(
    local_pool: &KvPool,
    global_pool: &KvPool,
    kv: &mut GemmaKv,
    new_tokens: usize,
) -> Result<()> {
    anyhow::ensure!(
        kv.local.belongs_to(local_pool) && kv.global.belongs_to(global_pool),
        "this KV state was allocated from different pools; admitting against \
         these would hand out page ids the executor cannot address"
    );
    let kv_len = kv.local.seq_len() + new_tokens;
    anyhow::ensure!(
        kv.local.seq_len() == kv.global.seq_len(),
        "the two families' frontiers diverged: local {} global {}",
        kv.local.seq_len(),
        kv.global.seq_len()
    );
    // Not saturating: a state past its frontier's account is a bookkeeping
    // error, and swallowing it defers the failure to the step that reads the
    // surplus page.
    let mut need = [0usize; 2];
    for (slot, (family, held, page_size)) in [
        (
            "local",
            kv.local.held_pages(),
            local_pool.layout().page_size,
        ),
        (
            "global",
            kv.global.held_pages(),
            global_pool.layout().page_size,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let want = kv_len.div_ceil(page_size);
        need[slot] = want.checked_sub(held).ok_or_else(|| {
            anyhow::anyhow!(
                "{family} family already holds {held} pages where {kv_len} \
                 tokens account for {want}"
            )
        })?;
    }
    let (local_need, global_need) = (need[0], need[1]);
    match (
        local_pool.try_reserve(local_need),
        global_pool.try_reserve(global_need),
    ) {
        (Some(local_r), Some(global_r)) => {
            kv.local.commit_reservation(local_r);
            kv.global.commit_reservation(global_r);
            Ok(())
        }
        (local_r, global_r) => {
            let (local_granted, global_granted) = (local_r.is_some(), global_r.is_some());
            // Report availability after rollback.
            drop((local_r, global_r));
            anyhow::bail!(
                "admission refused for {new_tokens} tokens (kv_len {kv_len}): \
                 local need {local_need} avail {} ({}), global need {global_need} avail {} ({})",
                local_pool.available_pages(),
                if local_granted {
                    "granted, rolled back"
                } else {
                    "refused"
                },
                global_pool.available_pages(),
                if global_granted {
                    "granted, rolled back"
                } else {
                    "refused"
                },
            )
        }
    }
}

pub(crate) const PAGE_SIZE: usize = 16;

#[cfg(test)]
mod tests {
    use pegainfer_core::tensor::DeviceContext;

    use super::*;

    fn tiny_pools(ctx: &DeviceContext) -> (KvPool, KvPool) {
        // 1 layer, 1 head, dim 1: just enough to exercise page accounting.
        // Capacities include the padding page each pool reserves: 4 -> 3
        // usable, 2 -> 1 usable.
        let local = KvPool::new(ctx, 1, 1, 1, 16, 4).expect("local pool");
        let global = KvPool::new(ctx, 1, 1, 1, 16, 2).expect("global pool");
        (local, global)
    }

    fn kv_from(local: &KvPool, global: &KvPool) -> GemmaKv {
        GemmaKv {
            local: local.alloc(),
            global: global.alloc(),
            local_resident_origin: 0,
        }
    }

    #[test]
    fn admission_is_atomic_across_pools() {
        let ctx = DeviceContext::new().expect("GPU required");
        let (local, global) = tiny_pools(&ctx);
        let mut kv = kv_from(&local, &global);

        // 17 tokens: local needs 2 of 3 (grantable), global needs 2 of 1
        // (refused). The grantable half must roll back.
        let refused = admit_tokens(&local, &global, &mut kv, 17);
        assert!(refused.is_err(), "partial admission must refuse");
        assert_eq!(local.available_pages(), 3, "local occupancy must roll back");
        assert_eq!(global.available_pages(), 1, "global occupancy untouched");
        assert_eq!(kv.local.held_pages(), 0);
        assert_eq!(kv.global.held_pages(), 0);

        admit_tokens(&local, &global, &mut kv, 16).expect("one page each");
        assert_eq!((local.available_pages(), global.available_pages()), (2, 0));
        assert_eq!((kv.local.held_pages(), kv.global.held_pages()), (1, 1));
    }
}
