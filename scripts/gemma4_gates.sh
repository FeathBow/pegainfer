#!/usr/bin/env bash
# Maintainer runner for the Gemma 4 checkpoint-backed gates.
#
# The gates below need a real checkpoint, its fixtures and a device, so CI
# only compiles them. This script owns their execution: it refuses to start
# unless every prerequisite is present, it holds the discovered gate set
# against the manifest here so a gate cannot quietly leave the suite. It
# claims one physical device for the suite's lifetime and runs one gate per
# process — repeated 12B loads in one test binary exhaust a 48 GiB card.
#
#   PEGAINFER_TEST_MODEL_PATH=<dense-checkpoint> \
#     PEGAINFER_NVFP4_MODEL=<routed-checkpoint> \
#     [PEGAINFER_GATE_GPU=<index-or-UUID>] scripts/gemma4_gates.sh [filter]
#
# A filter runs the subset of manifest gates whose names contain it; the
# membership check still covers the whole manifest.
set -uo pipefail

CRATE=pegainfer-gemma4
FEATURE=gemma4
GPU_LOCK_ROOT=/tmp

# The maintained suite, grouped by the invariant each gate owns. Adding a
# gate to the crate without adding it here fails the membership check, which
# covers every target an ignored test can live in — the library and each
# integration binary — so a gate cannot hide in one the runner never looks at.
#
# Each entry is "<needs> <gate>". `needs` is a comma-separated subset of
#   gpu         a CUDA device
#   ckpt        PEGAINFER_TEST_MODEL_PATH and the config it holds
#   moeckpt     PEGAINFER_NVFP4_MODEL and the routed config it holds
#   prompts     the generate fixture, read for its prompts only
#   fixtures    all four tensor fixtures, held against the checkpoint's digests
#   chatgolden  the chat/tokenizer reference JSON
# and a run demands only the union over the gates it selects, so a filter can
# run a gate without producing the whole suite's prerequisites.
GATES_NUMERIC_PARITY=(
  "gpu,ckpt,fixtures serve::oracle::context_waypoints_match_hf"
  "gpu,ckpt,fixtures serve::oracle::greedy_matches_hf_generate"
)
GATES_ADMISSION=(
  "gpu,ckpt,prompts serve::oracle::mixed_step_matches_serial"
  "gpu,ckpt,prompts engine::lane_tests::the_gathered_walk_matches_the_serial_path"
  "gpu,ckpt,prompts engine::lane_tests::the_gathered_transient_leaves_headroom"
)
# The roster-edge gate borrows the generate fixture's prompts too; the raise
# refusals are settled by `EngineState::load` before it opens a device or
# reads a weight, so that one needs the config and nothing else.
GATES_SERVING_CONTRACT=(
  "gpu,ckpt engine::lane_tests::the_shared_lane_lifecycle_completes"
  "gpu,ckpt engine::lane_tests::the_green_lane_lifecycle_completes"
  "gpu,ckpt engine::lane_tests::the_gathered_lifecycle_completes"
  "gpu,ckpt,prompts engine::lane_tests::the_raised_ceiling_and_slots_hold_at_the_roster_edge"
  "gpu,ckpt engine::lane_tests::the_raise_reaches_the_frontend"
  "ckpt engine::lane_tests::the_raise_refuses_without_its_prerequisites"
)
GATES_KV_AND_LANES=(
  "gpu,ckpt serve::oracle::eviction_is_footprint_only"
  "gpu,ckpt serve::oracle::prefix_restore_matches_cold_path"
  "gpu,ckpt serve::oracle::overlapped_prefill_matches_the_sync_step"
  "gpu,ckpt serve::oracle::a_ragged_batch_does_not_depend_on_row_order"
)
# The disagreeing-config gate deliberately fails before any device is opened.
GATES_LOADER=(
  "gpu,ckpt weights::load::tests::loads_the_text_tower_and_reports_residency"
  "ckpt weights::load::tests::a_disagreeing_config_names_every_faulty_tensor"
)
GATES_DEVICE=(
  "gpu engine::gate::the_suppression_mask_writes_only_the_ids_it_is_given"
  "gpu kv::tests::admission_is_atomic_across_pools"
)
GATES_ROUTED=(
  "gpu,moeckpt moe::tests::the_routed_block_matches_the_reference_formulas"
)
MANIFEST_LIB=(
  "${GATES_NUMERIC_PARITY[@]}"
  "${GATES_ADMISSION[@]}"
  "${GATES_SERVING_CONTRACT[@]}"
  "${GATES_KV_AND_LANES[@]}"
  "${GATES_LOADER[@]}"
  "${GATES_DEVICE[@]}"
  "${GATES_ROUTED[@]}"
)

# Integration gates live in their own binaries, which `--lib` cannot see. One
# array per binary, named GATES_<TARGET>; the target list itself is held
# against `tests/*.rs` below, so a new binary fails the check rather than
# going unowned.
INTEGRATION_TARGETS=(tokenizer_parity)
# Expanded indirectly from each name in INTEGRATION_TARGETS.
# shellcheck disable=SC2034
GATES_TOKENIZER_PARITY=(
  "ckpt,chatgolden string_form_chat_renders_match_hf_reference"
)

FIXTURES=(
  test_data/gemma4-12b-hf-golden.safetensors
  test_data/gemma4-12b-hf-window-golden.safetensors
  test_data/gemma4-12b-hf-longctx-golden.safetensors
  test_data/gemma4-12b-generate.safetensors
)
PROMPT_FIXTURE=test_data/gemma4-12b-generate.safetensors
CHAT_GOLDEN=test_data/gemma4-tokenizer-golden.json

die() { echo "gemma4 gates: $*" >&2; exit 1; }

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root" || die "cannot enter the repository root"

# --- prerequisites, one refusal per tier ----------------------------------
# Each is demanded only when a selected gate declares it, so a focused run
# carries the cost of what it runs: the device-only gates need no checkpoint,
# and the chat-render gate needs no card.
ckpt=
moe_ckpt=
gpu_uuid=
gpu_lock_fd=

require_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable, so no device can be claimed"
  command -v flock >/dev/null 2>&1 || die "flock is unavailable, so device ownership cannot be enforced"

  local selector=${PEGAINFER_GATE_GPU:-}
  if [ -z "$selector" ] && [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
    [ -n "$CUDA_VISIBLE_DEVICES" ] || die \
      "CUDA_VISIBLE_DEVICES is empty; set PEGAINFER_GATE_GPU to claim a device"
    [[ $CUDA_VISIBLE_DEVICES != *,* ]] || die \
      "CUDA_VISIBLE_DEVICES must name one device; set PEGAINFER_GATE_GPU explicitly"
    selector=$CUDA_VISIBLE_DEVICES
  fi
  selector=${selector:-0}
  [[ $selector != *,* ]] || die "PEGAINFER_GATE_GPU must name exactly one device"

  local rows=() row compute_mode lock_key lock_path
  mapfile -t rows < <(
    nvidia-smi -i "$selector" --query-gpu=uuid,compute_mode --format=csv,noheader 2>/dev/null
  )
  [ ${#rows[@]} -eq 1 ] || die "device selector $selector does not resolve to one GPU"
  row=${rows[0]}
  gpu_uuid=${row%%,*}
  gpu_uuid=${gpu_uuid//[[:space:]]/}
  compute_mode=${row#*,}
  compute_mode=${compute_mode#"${compute_mode%%[![:space:]]*}"}
  compute_mode=${compute_mode%"${compute_mode##*[![:space:]]}"}
  [ "$compute_mode" != Prohibited ] || die "GPU $gpu_uuid prohibits compute contexts"
  [[ $gpu_uuid =~ ^[A-Za-z0-9._:/-]+$ ]] || die "nvidia-smi returned an unsafe GPU identity"

  export CUDA_VISIBLE_DEVICES=$gpu_uuid
  lock_key=${gpu_uuid//\//_}
  lock_key=${lock_key//:/_}
  lock_path=$GPU_LOCK_ROOT/pegainfer-gemma4-gates-$lock_key.lock
  if (umask 022; set -o noclobber; : >"$lock_path") 2>/dev/null; then
    :
  elif [ ! -e "$lock_path" ]; then
    die "cannot create device lock $lock_path"
  fi
  # A read-only descriptor lets separate Unix accounts lock the same inode.
  exec {gpu_lock_fd}<"$lock_path" || die "cannot open device lock $lock_path"
  flock -n "$gpu_lock_fd" || die "GPU $gpu_uuid is already owned by another Gemma 4 gate runner"
  echo "gemma4 gates: claimed GPU $gpu_uuid (selector $selector)"
}

require_ckpt() {
  [ -n "${PEGAINFER_TEST_MODEL_PATH:-}" ] || die "PEGAINFER_TEST_MODEL_PATH is unset"
  ckpt=$PEGAINFER_TEST_MODEL_PATH
  [ -d "$ckpt" ] || die "checkpoint directory $ckpt does not exist"
  [ -f "$ckpt/config.json" ] || die "$ckpt has no config.json"
}

require_moeckpt() {
  [ -n "${PEGAINFER_NVFP4_MODEL:-}" ] || die "PEGAINFER_NVFP4_MODEL is unset"
  moe_ckpt=$PEGAINFER_NVFP4_MODEL
  [ -d "$moe_ckpt" ] || die "routed checkpoint directory $moe_ckpt does not exist"
  [ -f "$moe_ckpt/config.json" ] || die "$moe_ckpt has no config.json"
}

require_prompts() {
  [ -f "$PROMPT_FIXTURE" ] || die "fixture $PROMPT_FIXTURE is missing (dump it on the test box first)"
}

require_chatgolden() {
  [ -f "$CHAT_GOLDEN" ] || die "reference $CHAT_GOLDEN is missing (dump it on the test box first)"
}

require_fixtures() {
  require_ckpt
  local fixture
  for fixture in "${FIXTURES[@]}"; do
    [ -f "$fixture" ] || die "fixture $fixture is missing (dump it on the test box first)"
  done
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
}


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
lib_names=()
for entry in "${MANIFEST_LIB[@]}"; do lib_names+=("${entry##* }"); done
check_membership "library" "$lib_listing" "$(printf '%s\n' "${lib_names[@]}" | sort)"

# The integration binaries the crate actually has, so adding one without a
# manifest entry fails here instead of leaving its gates unowned.
discovered=$(find "$CRATE/tests" -maxdepth 1 -name '*.rs' -exec basename {} .rs \; 2>/dev/null | sort)
declared=$(printf '%s\n' "${INTEGRATION_TARGETS[@]}" | sort)
[ "$discovered" = "$declared" ] || die \
  "integration binaries disagree with INTEGRATION_TARGETS:"$'\n'"on disk: $discovered"$'\n'"declared: $declared"

all_gates=()
for entry in "${MANIFEST_LIB[@]}"; do all_gates+=("${entry%% *}|lib|${entry##* }"); done
for target in "${INTEGRATION_TARGETS[@]}"; do
  group="GATES_$(printf '%s' "$target" | tr '[:lower:]' '[:upper:]')[@]"
  target_names=()
  for entry in "${!group}"; do target_names+=("${entry##* }"); done
  check_membership "integration binary $target" \
    "$(ignored_in --test "$target")" "$(printf '%s\n' "${target_names[@]}" | sort)"
  for entry in "${!group}"; do all_gates+=("${entry%% *}|$target|${entry##* }"); done
done

filter=${1:-}
selected=()
for entry in "${all_gates[@]}"; do
  [ -z "$filter" ] || [[ ${entry##*|} == *"$filter"* ]] || continue
  selected+=("$entry")
done
[ ${#selected[@]} -gt 0 ] || die "filter ${filter:-<none>} selected no gate"

# --- prerequisites: the union over what this run selected, and no more -----
needs=" "
for entry in "${selected[@]}"; do needs="$needs${entry%%|*} "; done
needs=" ${needs//,/ } "
demanded=""
for want in gpu ckpt moeckpt prompts fixtures chatgolden; do
  case "$needs" in *" $want "*) "require_$want"; demanded="$demanded $want" ;; esac
done
echo "gemma4 gates: prerequisites$demanded"

echo "gemma4 gates: source $(git rev-parse HEAD)$([ -n "$(git status --porcelain)" ] && echo ' (dirty)')"
[ -z "$ckpt" ] || echo "gemma4 gates: checkpoint $ckpt"
[ -z "$moe_ckpt" ] || echo "gemma4 gates: routed checkpoint $moe_ckpt"
echo "gemma4 gates: ${#selected[@]} selected of ${#all_gates[@]} in the manifest"
printf '  %s\n' "${selected[@]##*|}"

# --- execution: one gate per process, serialized --------------------------
completed=0
failed=()
for entry in "${selected[@]}"; do
  IFS='|' read -r _needs target gate <<<"$entry"
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
