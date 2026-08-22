#!/usr/bin/env bash
# Maintainer runner for the Gemma 4 checkpoint-backed gates.
#
# The gates below need a real checkpoint, its fixtures and a device, so CI
# only compiles them. This script owns their execution: it refuses to start
# unless every prerequisite is present, it holds the discovered gate set
# against the manifest here so a gate cannot quietly leave the suite, and it
# runs one gate per process so two checkpoint-heavy tests never share a
# device — repeated 12B loads in one test binary exhaust a 48 GiB card.
#
#   PEGAINFER_TEST_MODEL_PATH=<checkpoint> scripts/gemma4_gates.sh [filter]
#
# A filter runs the subset of manifest gates whose names contain it; the
# membership check still covers the whole manifest.
set -uo pipefail

CRATE=pegainfer-gemma4
FEATURE=gemma4

# The maintained suite, grouped by the invariant each gate owns. Adding a
# gate to the crate without adding it here fails the membership check, which
# covers every target an ignored test can live in — the library and each
# integration binary — so a gate cannot hide in one the runner never looks at.
GATES_NUMERIC_PARITY=(
  serve::oracle::context_waypoints_match_hf
  serve::oracle::greedy_matches_hf_generate
)
GATES_ADMISSION=(
  serve::oracle::mixed_step_matches_serial
  engine::lane_tests::the_gathered_walk_matches_the_serial_path
  engine::lane_tests::the_gathered_transient_leaves_headroom
)
GATES_SERVING_CONTRACT=(
  engine::lane_tests::the_shared_lane_lifecycle_completes
  engine::lane_tests::the_green_lane_lifecycle_completes
  engine::lane_tests::the_gathered_lifecycle_completes
  engine::lane_tests::the_raised_ceiling_and_slots_hold_at_the_roster_edge
  engine::lane_tests::the_raise_reaches_the_frontend
  engine::lane_tests::the_raise_refuses_without_its_prerequisites
)
GATES_KV_AND_LANES=(
  serve::oracle::eviction_is_footprint_only
  serve::oracle::prefix_restore_matches_cold_path
  serve::oracle::overlapped_prefill_matches_the_sync_step
  serve::oracle::a_ragged_batch_does_not_depend_on_row_order
)
GATES_LOADER=(
  weights::load::tests::loads_the_text_tower_and_reports_residency
  weights::load::tests::a_disagreeing_config_names_every_faulty_tensor
)
# A card but no checkpoint is all these two need. They run under the same
# preflight regardless: a gate the runner does not own is a gate nothing runs.
GATES_DEVICE=(
  engine::gate::the_suppression_mask_writes_only_the_ids_it_is_given
  kv::tests::admission_is_atomic_across_pools
)
MANIFEST_LIB=(
  "${GATES_NUMERIC_PARITY[@]}"
  "${GATES_ADMISSION[@]}"
  "${GATES_SERVING_CONTRACT[@]}"
  "${GATES_KV_AND_LANES[@]}"
  "${GATES_LOADER[@]}"
  "${GATES_DEVICE[@]}"
)

# Integration gates live in their own binaries, which `--lib` cannot see. One
# array per binary, named GATES_<TARGET>; the target list itself is held
# against `tests/*.rs` below, so a new binary fails the check rather than
# going unowned.
INTEGRATION_TARGETS=(tokenizer_parity)
GATES_TOKENIZER_PARITY=(
  string_form_chat_renders_match_hf_reference
)

FIXTURES=(
  test_data/gemma4-12b-hf-golden.safetensors
  test_data/gemma4-12b-hf-window-golden.safetensors
  test_data/gemma4-12b-hf-longctx-golden.safetensors
  test_data/gemma4-12b-generate.safetensors
)

die() { echo "gemma4 gates: $*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root" || die "cannot enter the repository root"

# --- preflight: every reason to refuse, before anything runs ---------------
[ -n "${PEGAINFER_TEST_MODEL_PATH:-}" ] || die "PEGAINFER_TEST_MODEL_PATH is unset"
ckpt=$PEGAINFER_TEST_MODEL_PATH
[ -d "$ckpt" ] || die "checkpoint directory $ckpt does not exist"
[ -f "$ckpt/config.json" ] || die "$ckpt has no config.json"
for fixture in "${FIXTURES[@]}"; do
  [ -f "$fixture" ] || die "fixture $fixture is missing (dump it on the test box first)"
done
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable, so no device can be claimed"
nvidia-smi -L 2>/dev/null | grep -q GPU || die "nvidia-smi lists no device"

# The fixtures pin the checkpoint they were dumped from; the gates assert it
# per-run, but a mismatch should stop the suite before the first 12B load.
python3 - "$ckpt" "${FIXTURES[@]}" <<'PY' || die "fixture metadata preflight failed"
import hashlib, json, os, struct, sys

ckpt, fixtures = sys.argv[1], sys.argv[2:]

def manifest(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n)).get("__metadata__") or {}
    if len(meta) != 1:
        raise SystemExit(f"{path}: expected exactly one metadata key, found {sorted(meta)}")
    return json.loads(next(iter(meta.values())))

base = manifest(fixtures[0])
revision = base.get("revision")
if not revision:
    raise SystemExit(f"{fixtures[0]}: manifest carries no revision")
for path in fixtures[1:]:
    other = manifest(path).get("revision")
    if other != revision:
        raise SystemExit(f"{path}: revision {other} does not match {revision}")

digests = base.get("file_sha256") or {}
if not digests:
    raise SystemExit(f"{fixtures[0]}: manifest carries no file_sha256 block")
# The dumper's convention, mirrored from the crate's own checker: a
# "<file>#header" key hashes the safetensors header alone — the tensor
# layout without the 22 GiB payload — and a plain key hashes the whole file.
for name, want in digests.items():
    header_only = name.endswith("#header")
    filename = name[: -len("#header")] if header_only else name
    target = os.path.join(ckpt, filename)
    if not os.path.exists(target):
        raise SystemExit(f"checkpoint is missing {filename}")
    with open(target, "rb") as fh:
        if header_only:
            length = struct.unpack("<Q", fh.read(8))[0]
            payload = fh.read(length)
        else:
            payload = fh.read()
    if hashlib.sha256(payload).hexdigest() != want:
        raise SystemExit(f"{name}: checkpoint digest does not match the fixture's")
print(f"preflight: {len(fixtures)} fixtures agree on revision {revision[:12]}")
PY

# --- membership: the crate's ignored set must be exactly the manifest ------
ignored_in() {
  cargo test --release -p "$CRATE" --features "$FEATURE" "$@" -- \
    --ignored --list 2>/dev/null | sed -n 's/^\(.*\): test$/\1/p' | sort
}

check_membership() {
  local what=$1 listing=$2 expected=$3 missing extra
  missing=$(comm -13 <(printf '%s\n' "$listing") <(printf '%s\n' "$expected"))
  extra=$(comm -23 <(printf '%s\n' "$listing") <(printf '%s\n' "$expected"))
  [ -z "$missing" ] || die "the manifest names $what gates that do not exist:"$'\n'"$missing"
  [ -z "$extra" ] || die "$what has ignored gates the manifest does not name:"$'\n'"$extra"
}

lib_listing=$(ignored_in --lib)
[ -n "$lib_listing" ] || die "could not list the library's ignored gates"
check_membership "library" "$lib_listing" "$(printf '%s\n' "${MANIFEST_LIB[@]}" | sort)"

# The integration binaries the crate actually has, so adding one without a
# manifest entry fails here instead of leaving its gates unowned.
discovered=$(find "$CRATE/tests" -maxdepth 1 -name '*.rs' -exec basename {} .rs \; 2>/dev/null | sort)
declared=$(printf '%s\n' "${INTEGRATION_TARGETS[@]}" | sort)
[ "$discovered" = "$declared" ] || die \
  "integration binaries disagree with INTEGRATION_TARGETS:"$'\n'"on disk: $discovered"$'\n'"declared: $declared"

all_gates=()
for gate in "${MANIFEST_LIB[@]}"; do all_gates+=("lib|$gate"); done
for target in "${INTEGRATION_TARGETS[@]}"; do
  names="GATES_$(printf '%s' "$target" | tr '[:lower:]' '[:upper:]')[@]"
  check_membership "integration binary $target" \
    "$(ignored_in --test "$target")" "$(printf '%s\n' "${!names}" | sort)"
  for gate in "${!names}"; do all_gates+=("$target|$gate"); done
done

filter=${1:-}
selected=()
for entry in "${all_gates[@]}"; do
  [ -z "$filter" ] || [[ ${entry#*|} == *"$filter"* ]] || continue
  selected+=("$entry")
done
[ ${#selected[@]} -gt 0 ] || die "filter ${filter:-<none>} selected no gate"

echo "gemma4 gates: source $(git rev-parse HEAD)$([ -n "$(git status --porcelain)" ] && echo ' (dirty)')"
echo "gemma4 gates: checkpoint $ckpt"
echo "gemma4 gates: ${#selected[@]} selected of ${#all_gates[@]} in the manifest"
printf '  %s\n' "${selected[@]#*|}"

# --- execution: one gate per process, serialized --------------------------
completed=0
failed=()
for entry in "${selected[@]}"; do
  target=${entry%%|*}
  gate=${entry#*|}
  if [ "$target" = lib ]; then
    target_args=(--lib)
  else
    target_args=(--test "$target")
  fi
  echo "--- $gate"
  if cargo test --release -p "$CRATE" --features "$FEATURE" "${target_args[@]}" -- \
      --ignored --exact "$gate" --test-threads=1 --nocapture 2>&1 | tail -20; then
    completed=$((completed + 1))
  else
    failed+=("$gate")
  fi
done

echo "gemma4 gates: selected ${#selected[@]}, completed $completed, failed ${#failed[@]}"
if [ ${#failed[@]} -gt 0 ]; then
  printf 'gemma4 gates: FAILED %s\n' "${failed[@]}"
  exit 1
fi
[ "$completed" -eq "${#selected[@]}" ] || die "a selected gate neither completed nor failed"
echo "gemma4 gates: all selected gates completed"
