"""AOT-compile the Gemma 4 hd512 prefill kernel into one CUDA file.

`pegainfer-kernels/build.rs` runs this under the `gemma4` feature and hands
the result to nvcc. It prints the same `KEY=VALUE` manifest every TileLang
family prints, and mirrors it into `manifest.txt` so a build host can consume
a pre-generated directory without TileLang installed.

What makes this family different from the K3 one is the lowering: the key and
query loads become bulk copies, so TileLang passes those two tensors as TMA
descriptors instead of pointers and adds a producer warpgroup to the block.
The descriptors are the launcher's to build, and every parameter it needs is
a constant in the lowered host stub — recovered here rather than assumed, so
a codegen change fails the build instead of encoding a stale descriptor.

The declared tensor extents are the serving arena's maxima. They reach the
device code only as bounds guards, so any smaller step passes them; the two
TMA tensors carry their real extents in the descriptors, which the launcher
builds per call from the arguments it is given.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import tilelang
import tilelang_defs as defs
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

ENTRY_SYMBOL = "main_kernel"
KERNEL_MARKER = 'extern "C" __global__ void'
LAUNCHER = "gemma4_hd512_prefill_varlen"
CU_STEM = "gemma4_hd512_prefill"
# TileLang names every entry point `main_kernel` with external C linkage, so
# the emitted body is renamed before it can collide with another family's.
KERNEL_SYMBOL = f"{CU_STEM}_kernel"

# The launcher's C parameters, without the trailing stream the consumer adds.
# The emitted definition and the manifest line both come from here, so the
# stub the build script writes when this kernel is absent cannot drift from
# the real one: C has no mangling, and a drifted pair would link silently.
LAUNCHER_PARAMS = [
    ("const void*", "q"),
    ("const void*", "kv"),
    ("const int*", "page_indices"),
    ("const int*", "page_indptr"),
    ("const int*", "q_indptr"),
    ("const int*", "host_q_indptr"),
    ("const int*", "last_page_len"),
    ("void*", "out"),
    ("int", "batch"),
    ("int", "q_rows"),
    ("int", "pool_rows"),
    ("int", "rows_per_page"),
    ("int", "layer_row"),
    ("int", "page_size"),
    ("int", "num_qo_heads"),
    ("int", "num_kv_heads"),
    ("float", "sm_scale"),
]

# TileLang's `debug.h` *defines* `debug_print_msg` and the `uint16_t`
# specialization of `debug_print_buffer_value` with external linkage, so every
# translation unit that includes it exports the same two symbols. This kernel
# calls neither, and a binary that also links a K3 family would get duplicate
# definitions, so this unit is given privately named copies.
DEBUG_HEADER = "#include <tl_templates/cuda/debug.h>"
DEBUG_HELPERS = ("debug_print_msg", "debug_print_buffer_value")

# The model's global-attention geometry. Shapes are compile dimensions, so
# these are the kernel's identity, not run-time inputs.
HEADS = 32
GROUPS = 8
HEAD_DIM = 512
PAGE_SIZE = 64

# The serving arena's maxima, in the units each tensor is indexed in. A step
# is always smaller; see the module docstring.
MAX_BATCH = 8
CEILING = 262144
SLOTS = 16
Q_ROWS = CEILING + defs.BLOCK_M
POOL_PAGES = SLOTS * (CEILING // PAGE_SIZE) + 1
PAGE_TABLE_LEN = SLOTS * (CEILING // PAGE_SIZE)
# The pool is addressed in rows of [kv_heads, head_dim]; one page holds every
# layer's K and V, and the deepest checkpoint this line serves sets the bound.
MAX_LAYERS = 64
POOL_ROWS = POOL_PAGES * MAX_LAYERS * 2 * PAGE_SIZE
# The launcher takes both row counts as C ints; a configuration that outgrew
# one would index past the guards rather than fail.
assert Q_ROWS < 2**31, Q_ROWS
assert POOL_ROWS < 2**31, POOL_ROWS

# Past 48 KiB a kernel has to opt into its dynamic shared memory per symbol.
MAX_STATIC_SMEM = 48 * 1024
# SM90's per-block ceiling. A recovered size above it would launch-fail.
MAX_DYNAMIC_SMEM = 227 * 1024

TENSORMAP_BUILDER = "__tvm_tensormap_create_tiled"

# A pass config can reach nvcc's command line, which TileLang's JIT passes and
# an AOT build would not: without --use_fast_math the object sits half a ULP
# from the gated kernel. Every config is classified here or generation stops.
PASS_CONFIG_NVCC_FLAG = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: "--use_fast_math"}

# TVM's tensormap codes, mapped to the driver enums the launcher passes. Every
# code the stub can carry is listed; an unmapped one is a codegen change.
TENSORMAP_DTYPE = {9: "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16"}
TENSORMAP_INTERLEAVE = {0: "CU_TENSOR_MAP_INTERLEAVE_NONE"}
TENSORMAP_SWIZZLE = {
    0: "CU_TENSOR_MAP_SWIZZLE_NONE",
    1: "CU_TENSOR_MAP_SWIZZLE_32B",
    2: "CU_TENSOR_MAP_SWIZZLE_64B",
    3: "CU_TENSOR_MAP_SWIZZLE_128B",
}
TENSORMAP_L2 = {
    0: "CU_TENSOR_MAP_L2_PROMOTION_NONE",
    1: "CU_TENSOR_MAP_L2_PROMOTION_L2_64B",
    2: "CU_TENSOR_MAP_L2_PROMOTION_L2_128B",
    3: "CU_TENSOR_MAP_L2_PROMOTION_L2_256B",
}
TENSORMAP_OOB = {0: "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"}

_SLOT_INT = re.compile(
    r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_int64\) = \(\(int64_t\)(-?\d+)\);"
)
_SLOT_PTR = re.compile(r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_ptr\) = (\w+);")
_PACKED_CALL = re.compile(
    r"TVMFFIFunctionCall\((\w+?)_packed, \(TVMFFIAny\*\) stack_ffi_any, (\d+),"
)


@dataclass(frozen=True)
class TensorMap:
    """One recovered `cuTensorMapEncodeTiled` call, still in TVM's spelling."""

    name: str
    dtype: int
    rank: int
    tensor: str
    dims: tuple[int, ...]
    strides: tuple[int, ...]
    box: tuple[int, ...]
    element_strides: tuple[int, ...]
    interleave: int
    swizzle: int
    l2_promotion: int
    oob_fill: int


def read_host_stub(kernel) -> tuple[list[TensorMap], int, int]:
    """Recover the descriptor builds, the block width and the dynamic smem.

    TileLang bakes the launch into the host stub as a packed-call argument
    stack and exposes no accessor for it, hence the parse. Slots persist
    across calls, so only the values a call writes itself are its own; the
    entry call's grid and scalars are run-time expressions and are read from
    its arguments instead, which is why they never appear here.
    """
    slots: dict[int, object] = {}
    maps: list[TensorMap] = []
    launch: list[object] | None = None
    for line in kernel.get_host_source().splitlines():
        match = _SLOT_INT.search(line)
        if match:
            slots[int(match.group(1))] = int(match.group(2))
            continue
        match = _SLOT_PTR.search(line)
        if match:
            slots[int(match.group(1))] = match.group(2)
            continue
        match = _PACKED_CALL.search(line)
        if not match:
            continue
        callee, count = match.group(1), int(match.group(2))
        args = [slots.get(i) for i in range(count)]
        if callee == TENSORMAP_BUILDER:
            maps.append(parse_tensormap(args))
        elif callee == ENTRY_SYMBOL:
            launch = args

    if not maps:
        raise RuntimeError(
            "the lowering built no TMA descriptor; this launcher exists only "
            "because it does, so the parameter list it binds is wrong now"
        )
    if launch is None:
        raise RuntimeError("could not recover the entry call from the host stub")
    # Block x/y/z then the dynamic smem, which TileLang omits when it is zero;
    # this kernel's four shared buffers make that impossible, so a tail without
    # the block's unit y and z is a codegen change.
    tail = list(launch[-4:])
    if tail[1:3] != [1, 1]:
        raise RuntimeError(
            f"the launch tail {tail} is not (block, 1, 1, dynamic shared); the "
            "lowering changed its geometry or dropped its shared memory"
        )
    block, smem = tail[0], tail[3]
    if not isinstance(block, int) or not isinstance(smem, int):
        raise TypeError(f"launch geometry has non-constant entries: {tail}")
    if smem > MAX_DYNAMIC_SMEM:
        raise RuntimeError(
            f"the lowering wants {smem} B of dynamic shared memory, past the "
            f"{MAX_DYNAMIC_SMEM} B a block can be given"
        )
    return maps, block, smem


def parse_tensormap(args: list) -> TensorMap:
    """Split one builder call's flat argument list by its recovered rank."""
    name, dtype, rank, tensor = args[0], args[1], args[2], args[3]
    if not isinstance(rank, int) or rank < 2:
        raise RuntimeError(f"descriptor {name} has a non-constant rank: {rank}")
    at = 4
    dims = tuple(args[at : at + rank])
    at += rank
    # The builder carries one stride per dimension, innermost first, and the
    # innermost one is the element size. The driver API takes the rest.
    strides = tuple(args[at : at + rank])
    at += rank
    box = tuple(args[at : at + rank])
    at += rank
    element_strides = tuple(args[at : at + rank])
    at += rank
    interleave, swizzle, l2_promotion, oob_fill = args[at : at + 4]
    values = (
        *dims,
        *strides,
        *box,
        *element_strides,
        interleave,
        swizzle,
        l2_promotion,
        oob_fill,
    )
    if any(not isinstance(value, int) for value in values):
        raise RuntimeError(f"descriptor {name} has non-constant parameters: {values}")
    return TensorMap(
        name=str(name),
        dtype=dtype,
        rank=rank,
        tensor=str(tensor),
        dims=dims,
        strides=strides,
        box=box,
        element_strides=element_strides,
        interleave=interleave,
        swizzle=swizzle,
        l2_promotion=l2_promotion,
        oob_fill=oob_fill,
    )


def enum_of(table: dict[int, str], code: int, what: str) -> str:
    if code not in table:
        raise RuntimeError(f"the lowering asked for an unmapped {what}: {code}")
    return table[code]


def runtime_dim(tmap: TensorMap, declared: int, label: str) -> int:
    """Which of a descriptor's dimensions the launcher fills in per call.

    The rows of a packed q buffer and of the pool are run-time quantities, and
    they enter the device code only through the descriptor, so the launcher
    substitutes them here. Matching on the declared value keeps the position
    tied to the lowering rather than to a hand-kept index.
    """
    hits = [i for i, dim in enumerate(tmap.dims) if dim == declared]
    if len(hits) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one dimension equal to {declared} in "
            f"{tmap.dims}, found {len(hits)}"
        )
    return hits[0]


# The launcher's own name for each thing the stub names. A descriptor the
# lowering grew, or a tensor it renamed, has no binding here and fails
# generation rather than emitting a launcher that does not compile.
DESCRIPTOR_VAR = {"Q_desc": "q_desc", "KV_desc": "kv_desc"}
TENSOR_ARG = {"Q": "q", "KV": "kv"}


def render_descriptor(tmap: TensorMap, rows_expr: str, rows_at: int) -> str:
    """The launcher body that encodes one descriptor."""
    if tmap.name not in DESCRIPTOR_VAR:
        raise RuntimeError(f"no launcher variable for descriptor {tmap.name}")
    if tmap.tensor not in TENSOR_ARG:
        raise RuntimeError(f"no launcher argument for tensor {tmap.tensor}")
    dims = [str(dim) for dim in tmap.dims]
    dims[rows_at] = rows_expr
    return (
        f"  {{\n"
        f"    const cuuint64_t dims[{tmap.rank}] = {{{', '.join(dims)}}};\n"
        f"    const cuuint64_t strides[{tmap.rank - 1}] = "
        f"{{{', '.join(str(s) for s in tmap.strides[1:])}}};\n"
        f"    const cuuint32_t box[{tmap.rank}] = "
        f"{{{', '.join(str(b) for b in tmap.box)}}};\n"
        f"    const cuuint32_t element_strides[{tmap.rank}] = "
        f"{{{', '.join(str(e) for e in tmap.element_strides)}}};\n"
        f"    const CUresult encoded = encode(\n"
        f"        &{DESCRIPTOR_VAR[tmap.name]}, "
        f"{enum_of(TENSORMAP_DTYPE, tmap.dtype, 'dtype')}, {tmap.rank},\n"
        f"        const_cast<void*>({TENSOR_ARG[tmap.tensor]}), dims, strides, box, "
        f"element_strides,\n"
        f"        {enum_of(TENSORMAP_INTERLEAVE, tmap.interleave, 'interleave')},\n"
        f"        {enum_of(TENSORMAP_SWIZZLE, tmap.swizzle, 'swizzle')},\n"
        f"        {enum_of(TENSORMAP_L2, tmap.l2_promotion, 'L2 promotion')},\n"
        f"        {enum_of(TENSORMAP_OOB, tmap.oob_fill, 'out-of-bounds fill')});\n"
        f"    if (encoded != CUDA_SUCCESS) {{\n"
        f"      return static_cast<int>(cudaErrorInvalidValue);\n"
        f"    }}\n"
        f"  }}\n"
    )


ELEMENT_BYTES = {"CU_TENSOR_MAP_DATA_TYPE_BFLOAT16": 2}

LAUNCHER_HEAD = """
// Hand-written launcher. The key and query loads lower to bulk copies, so the
// entry takes TMA descriptors for those two and plain pointers for the rest;
// every descriptor parameter below is recovered from the lowered host stub.
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// cuTensorMapEncodeTiled is a driver entry point, resolved once and cached.
CUresult encode(CUtensorMap* map, CUtensorMapDataType dtype, cuuint32_t rank,
                void* tensor, const cuuint64_t* dims, const cuuint64_t* strides,
                const cuuint32_t* box, const cuuint32_t* element_strides,
                CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                CUtensorMapL2promotion l2_promotion,
                CUtensorMapFloatOOBfill oob_fill) {
  using Fn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                          const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                          const cuuint32_t*, CUtensorMapInterleave,
                          CUtensorMapSwizzle, CUtensorMapL2promotion,
                          CUtensorMapFloatOOBfill);
  static Fn fn = [] {
    void* entry = nullptr;
    cudaDriverEntryPointQueryResult found;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &entry,
                                cudaEnableDefault, &found) != cudaSuccess ||
        found != cudaDriverEntryPointSuccess) {
      return static_cast<Fn>(nullptr);
    }
    return reinterpret_cast<Fn>(entry);
  }();
  if (fn == nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return fn(map, dtype, rank, tensor, dims, strides, box, element_strides,
            interleave, swizzle, l2_promotion, oob_fill);
}

}  // namespace
"""


def render_launcher(
    maps: list[TensorMap], block: int, smem: int, order: list[str]
) -> str:
    """The `extern "C"` entry `pegainfer-kernels` links against."""
    by_name = {tmap.name: tmap for tmap in maps}
    if set(by_name) != set(DESCRIPTOR_VAR):
        raise RuntimeError(
            f"expected descriptors {sorted(DESCRIPTOR_VAR)}, got {sorted(by_name)}"
        )

    bodies = [
        render_descriptor(
            by_name["Q_desc"],
            "static_cast<cuuint64_t>(q_rows)",
            runtime_dim(by_name["Q_desc"], Q_ROWS, "Q_desc"),
        ),
        render_descriptor(
            by_name["KV_desc"],
            "static_cast<cuuint64_t>(pool_rows)",
            runtime_dim(by_name["KV_desc"], POOL_ROWS, "KV_desc"),
        ),
    ]

    opt_in = ""
    if smem > MAX_STATIC_SMEM:
        opt_in = (
            f"  // Past 48 KiB the kernel has to opt in; once per symbol.\n"
            f"  static const cudaError_t opt_in = cudaFuncSetAttribute(\n"
            f"      reinterpret_cast<const void*>({KERNEL_SYMBOL}),\n"
            f"      cudaFuncAttributeMaxDynamicSharedMemorySize, {smem});\n"
            f"  if (opt_in != cudaSuccess) {{\n"
            f"    return static_cast<int>(opt_in);\n"
            f"  }}\n"
        )

    bound = {
        **DESCRIPTOR_VAR,
        "Output": "reinterpret_cast<bfloat16_t*>(out)",
        "PageIndices": "page_indices",
        "PageIndptr": "page_indptr",
        "QIndptr": "q_indptr",
        "LastPageLen": "last_page_len",
        "sm_scale": "sm_scale",
        "total_ctas": "total_ctas",
        "rows_per_page": "rows_per_page",
        "layer_row": "layer_row",
    }
    missing = [name for name in order if name not in bound]
    if missing:
        raise RuntimeError(
            f"the entry point grew parameters this launcher does not bind: {missing}"
        )
    args = ",\n      ".join(bound[name] for name in order)

    declarations = "".join(
        f"  alignas(64) CUtensorMap {var};\n" for var in DESCRIPTOR_VAR.values()
    )
    # Past any of these the body drops a tail or reads the wrong rows and hands
    # back plausible numbers, so the launcher refuses rather than truncates.
    bounds = (
        f"  if (q_rows > {Q_ROWS} || pool_rows > {POOL_ROWS} || batch > {MAX_BATCH}\n"
        f"      || page_size != {PAGE_SIZE} || num_qo_heads != {HEADS}\n"
        f"      || num_kv_heads != {HEADS // GROUPS}) {{\n"
        f"    return static_cast<int>(cudaErrorInvalidValue);\n"
        f"  }}\n"
    )
    # The grid is computed here, from the sum the body re-walks to find its
    # owner: a caller's own copy disagreeing is silent both ways, too large and
    # CTAs spin up to exit, too small and a request's tail never runs.
    grid = (
        f"  int total_ctas = 0;\n"
        f"  for (int request = 0; request < batch; ++request) {{\n"
        f"    const int rows = host_q_indptr[request + 1] - host_q_indptr[request];\n"
        f"    const int tiles = (rows + {defs.BLOCK_M - 1}) / {defs.BLOCK_M};\n"
        f"    total_ctas += ((tiles + {defs.QBLK - 1}) / {defs.QBLK})"
        f" * {defs.QBLK} * {HEADS};\n"
        f"  }}\n"
        f"  if (total_ctas == 0) {{\n"
        f"    // A step with no prompt rows is a real state, not an error.\n"
        f"    return static_cast<int>(cudaSuccess);\n"
        f"  }}\n"
    )
    signature = ", ".join(f"{kind} {name}" for kind, name in LAUNCHER_PARAMS)
    return (
        f'extern "C" int {LAUNCHER}(\n'
        f"    {signature},\n"
        f"    cudaStream_t stream) {{\n"
        f"{bounds}"
        f"{grid}"
        f"{declarations}"
        f"{''.join(bodies)}"
        f"{opt_in}"
        f"  {KERNEL_SYMBOL}<<<dim3(total_ctas), dim3({block}), {smem}, stream>>>(\n"
        f"      {args});\n"
        f"  return static_cast<int>(cudaGetLastError());\n"
        f"}}\n"
    )


def entry_parameter_order(source: str) -> list[str]:
    """Parameter names of the generated entry point, in its own order."""
    marker = source.index(f"{KERNEL_MARKER} {ENTRY_SYMBOL}(")
    open_at = source.index("(", marker)
    close_at = source.index(")", open_at)
    names = []
    for param in source[open_at + 1 : close_at].split(","):
        names.append(param.strip().split()[-1].lstrip("*"))
    return names


def split_source(source: str) -> tuple[str, str]:
    """Split `get_kernel_source()` into (include preamble, kernel bodies)."""
    marker = source.index(KERNEL_MARKER)
    return source[:marker], source[marker:]


def isolate_debug_helpers(preamble: str) -> str:
    """Rename `debug.h`'s externally linked helpers for this unit."""
    if DEBUG_HEADER not in preamble:
        return preamble
    renames = "".join(f"#define {name} {CU_STEM}_{name}\n" for name in DEBUG_HELPERS)
    restores = "".join(f"#undef {name}\n" for name in DEBUG_HELPERS)
    return preamble.replace(
        DEBUG_HEADER, f"{renames}{DEBUG_HEADER}\n{restores}".rstrip("\n")
    )


def vendor_includes(out_dir: Path) -> tuple[Path, Path]:
    """Copy the header roots in, so the directory stands on its own."""
    copied = []
    for source, name in (
        (TILELANG_TEMPLATE_PATH, "tilelang"),
        (CUTLASS_INCLUDE_DIR, "cutlass"),
    ):
        destination = out_dir / "include" / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        copied.append(destination)
    return copied[0], copied[1]


def required_nvcc_flags() -> list[str]:
    """The flags TileLang's JIT would pass for this kernel's pass configs."""
    flags = []
    for key, enabled in defs.PASS_CONFIGS.items():
        if key not in PASS_CONFIG_NVCC_FLAG:
            raise RuntimeError(
                f"pass config {key} is not classified: say whether it reaches "
                "nvcc's command line, or the generated object will not match "
                "the kernel that was gated"
            )
        flag = PASS_CONFIG_NVCC_FLAG[key]
        if enabled and flag is not None:
            flags.append(flag)
    return flags


def build_kernel(arch: str):
    """Lower for the arch the objects will be assembled for.

    Generation must not depend on a GPU being visible to the build host —
    containers routinely have none, and TileLang then lowers for its own
    default, which nvcc rejects outright. So the arch is always passed.
    """
    return tilelang.compile(
        defs.prefill_varlen(
            HEADS,
            GROUPS,
            HEAD_DIM,
            PAGE_SIZE,
            MAX_BATCH,
            Q_ROWS,
            POOL_ROWS,
            PAGE_TABLE_LEN,
        ),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def check_strides(maps: list[TensorMap]) -> None:
    """The builder's innermost stride is the element size; hold it to that."""
    for tmap in maps:
        dtype = enum_of(TENSORMAP_DTYPE, tmap.dtype, "dtype")
        if dtype not in ELEMENT_BYTES:
            raise RuntimeError(f"{dtype} has no element size to check against")
        expected = ELEMENT_BYTES[dtype]
        if tmap.strides[0] != expected:
            raise RuntimeError(
                f"{tmap.name}: innermost stride {tmap.strides[0]} is not the "
                f"{expected} B element of {dtype}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--arch",
        required=True,
        help="arch to lower and assemble for, e.g. sm_90a. Required: without "
        "it TileLang lowers for whatever device it can see, or for its own "
        "default on a host with none.",
    )
    parser.add_argument(
        "--vendor-includes",
        action="store_true",
        help="copy the header roots into the output and point the manifest at "
        "the copies (self-contained pre-generated dir)",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel = build_kernel(args.arch)
    source = kernel.get_kernel_source()
    maps, block, smem = read_host_stub(kernel)
    check_strides(maps)
    order = entry_parameter_order(source)
    preamble, body = split_source(source)
    if body.count(ENTRY_SYMBOL) != 2:
        raise RuntimeError(f"expected exactly two {ENTRY_SYMBOL} occurrences")

    cu_path = out_dir / f"{CU_STEM}.cu"
    cu_path.write_text(
        "// Generated by pegainfer-gemma4/kernels/generate.py. Do not edit.\n"
        + isolate_debug_helpers(preamble)
        + body.replace(ENTRY_SYMBOL, KERNEL_SYMBOL)
        + LAUNCHER_HEAD
        + render_launcher(maps, block, smem, order)
    )

    if args.vendor_includes:
        template_include, cutlass_include = vendor_includes(out_dir)
    else:
        template_include = Path(TILELANG_TEMPLATE_PATH)
        cutlass_include = Path(CUTLASS_INCLUDE_DIR)

    # Relative where it can be, so a vendored directory survives being copied.
    def named(path: Path) -> str:
        try:
            return str(path.relative_to(out_dir))
        except ValueError:
            return str(path)

    lines = [f"CU_PATH={named(cu_path)}"]
    lines.append(f"TILELANG_TEMPLATE_PATH={named(template_include)}")
    lines.append(f"CUTLASS_INCLUDE_DIR={named(cutlass_include)}")
    # Shapes are compile dimensions, so the consumer can refuse another.
    lines.append(f"GEOMETRY={HEADS},{HEADS // GROUPS},{HEAD_DIM},{PAGE_SIZE}")
    # The opt-in a block needs: SM90 grants 227 KiB, SM120 only 99.
    lines.append(f"SMEM={smem}")
    lines.extend(f"NVCC_FLAG={flag}" for flag in required_nvcc_flags())
    lines.append(
        f"LAUNCHER={LAUNCHER}|{', '.join(kind for kind, _ in LAUNCHER_PARAMS)}"
    )
    # The body is lowered for exactly this arch and uses arch-conditional
    # instructions, so the consumer assembles it for that and not for the
    # generic SM list.
    lines.append(f"ARCH={args.arch}")
    # The manifest lets a build host consume a pre-generated directory without
    # re-running (or even having) TileLang; build.rs parses the same key=value
    # lines from either stdout or this file.
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print(f"# block {block} threads, {smem} B dynamic shared, {len(maps)} descriptors")


if __name__ == "__main__":
    main()
