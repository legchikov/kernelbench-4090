"""AST-based static checks for candidate kernels.

KernelBench ships a regex checker (kernelbench.kernel_static_checker). This module works on the
syntax tree instead, so comments, strings and variable names don't confuse it, and it resolves
import aliases (``import torch.nn.functional as F`` -> ``F.relu`` is ``torch.nn.functional.relu``).
If KernelBench is importable its checker also runs, and its findings are reported as warnings.

Static checks are the first line of defence only. The runtime checks in evaluate.py are what
actually decide whether a kernel ran, ran on the right stream, and left the inputs alone.

Severity levels:
  error -> verdict INVALID (the candidate is not evaluated further)
  warn  -> recorded and shown, evaluation continues
  info  -> recorded for analysis (e.g. which torch ops the candidate still uses)
Override any rule's severity with ``policy.static.<rule>: error|warn|info|off`` in configs/rig.yaml.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

DEFAULT_SEVERITY: dict[str, str] = {
    "syntax_error": "error",
    "no_modelnew": "error",
    "no_kernel_backend": "error",
    "backend_not_allowed": "error",
    "subclasses_reference": "error",
    "instantiates_reference": "error",
    "kernel_defined_not_called": "error",
    "triton_kernel_not_launched": "warn",
    "torch_compile": "error",
    "numeric_flag_tamper": "error",
    "env_tamper": "error",
    "harness_tamper": "error",
    "introspection": "error",
    "dynamic_code": "error",
    "process_or_network": "error",
    "file_io": "warn",
    "threads": "error",
    "lazy_tensor": "error",
    "try_except": "error",
    "side_stream": "warn",
    "memoisation": "warn",
    "torch_ops": "info",
    "reduced_precision": "info",
    "kernelbench_checker": "warn",
}

# Library calls that are fine anywhere: building and loading extensions, parameters, containers.
_TORCH_NN_ALLOWED = {
    "Module", "Parameter", "ParameterList", "ParameterDict", "ModuleList", "ModuleDict", "init",
    "Buffer", "Identity",
}

_TORCH_COMPUTE_PREFIXES = (
    "torch.nn.functional.",
    "torch.matmul", "torch.mm", "torch.bmm", "torch.addmm", "torch.baddbmm", "torch.einsum",
    "torch.conv", "torch.softmax", "torch.log_softmax", "torch.relu", "torch.sigmoid", "torch.tanh",
    "torch.layer_norm", "torch.group_norm", "torch.batch_norm", "torch.instance_norm", "torch.lstm",
    "torch.gru", "torch.rnn", "torch.cumsum", "torch.cumprod", "torch.sum", "torch.mean",
    "torch.max", "torch.min", "torch.argmax", "torch.argmin", "torch.logsumexp", "torch.linear",
    "torch.nn.functional", "torch.ops.aten",
)

_NETWORK_OR_PROCESS = (
    "subprocess", "socket", "urllib", "requests", "http", "ftplib", "smtplib", "httpx", "aiohttp",
    "os.system", "os.popen", "os.spawn", "os.exec", "os.fork", "pty",
)

_THREAD_MODULES = ("threading", "multiprocessing", "concurrent", "asyncio", "_thread")

_INTROSPECTION = (
    "inspect.", "sys._getframe", "gc.get_objects", "gc.get_referrers", "gc.get_referents",
    "traceback.extract_stack", "traceback.walk_stack", "sys.settrace", "sys.setprofile",
)

_HARNESS_ROOTS = ("torch.", "time.", "builtins.", "sys.", "numpy.", "np.", "os.", "importlib.")

_CUDA_STREAM_PATTERNS = [
    r"cudaStreamCreate", r"getStreamFromPool", r"cublasSetStream", r"cudaStreamNonBlocking",
    r"cudaStreamPerThread", r"cudaStreamLegacy",
]

_REDUCED_PRECISION_PATTERNS = [
    r"__half\b", r"\bhalf2\b", r"__nv_bfloat16", r"__nv_fp8", r"\bwmma::", r"mma\.sync",
    r"tl\.float16", r"tl\.bfloat16", r"tl\.float8", r"allow_tf32\s*=\s*True",
    r"input_precision\s*=\s*['\"]tf32",
]


@dataclass
class Finding:
    rule: str
    severity: str
    message: str
    line: int | None = None


class _Collector(ast.NodeVisitor):
    """Walk the tree once and collect everything the rules need."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.imports: list[tuple[str, int]] = []
        self.calls: list[tuple[str, int, ast.Call]] = []
        self.assign_targets: list[tuple[str, int]] = []
        self.subscript_assigns: list[tuple[str, str | None, int]] = []
        self.classes: list[ast.ClassDef] = []
        self.functions: list[ast.FunctionDef] = []
        self.decorators: list[tuple[str, int, ast.AST]] = []
        self.strings: list[tuple[str, int]] = []
        self.attributes: list[tuple[str, int]] = []
        self.tries: list[int] = []
        self.subscript_calls: list[tuple[str, int]] = []
        self.class_stack: list[str] = []
        self.calls_in_modelnew: list[tuple[str, int]] = []

    # --- name resolution -------------------------------------------------------------------
    def dotted(self, node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Call):
            inner = self.dotted(node.func)
            if inner is None:
                return None
            parts.append(inner + "()")
        else:
            return None
        parts.reverse()
        head = self.aliases.get(parts[0], parts[0])
        return ".".join([head, *parts[1:]])

    # --- visitors --------------------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append((alias.name, node.lineno))
            if alias.asname:
                self.aliases[alias.asname] = alias.name
            else:
                self.aliases.setdefault(alias.name.split(".")[0], alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            full = f"{mod}.{alias.name}" if mod else alias.name
            self.imports.append((full, node.lineno))
            self.aliases[alias.asname or alias.name] = full
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node)
        for dec in node.decorator_list:
            name = self.dotted(dec.func if isinstance(dec, ast.Call) else dec)
            if name:
                self.decorators.append((name, node.lineno, dec))
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node)
        for dec in node.decorator_list:
            name = self.dotted(dec.func if isinstance(dec, ast.Call) else dec)
            if name:
                self.decorators.append((name, node.lineno, dec))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        name = self.dotted(node.func)
        if name:
            self.calls.append((name, node.lineno, node))
            if "ModelNew" in self.class_stack:
                self.calls_in_modelnew.append((name, node.lineno))
        if isinstance(node.func, ast.Subscript):
            target = self.dotted(node.func.value)
            if target:
                self.subscript_calls.append((target, node.lineno))
        self.generic_visit(node)

    def _record_target(self, target: ast.AST, lineno: int) -> None:
        if isinstance(target, ast.Attribute):
            name = self.dotted(target)
            if name:
                self.assign_targets.append((name, lineno))
        elif isinstance(target, ast.Subscript):
            base = self.dotted(target.value)
            key = None
            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                key = target.slice.value
            if base:
                self.subscript_assigns.append((base, key, lineno))
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self._record_target(elt, lineno)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_target(target, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_target(node.target, node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = self.dotted(node)
        if name:
            self.attributes.append((name, node.lineno))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.append((node.value, getattr(node, "lineno", 0)))

    def visit_Try(self, node: ast.Try) -> None:
        self.tries.append(node.lineno)
        self.generic_visit(node)

    visit_TryStar = visit_Try  # type: ignore[assignment]


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    """Plain prefix match: 'torch.max' also catches 'torch.max_pool2d', which is what we want."""
    return any(name.startswith(p) for p in prefixes)


def _load_inline_exports(call: ast.Call) -> list[str]:
    for kw in call.keywords:
        if kw.arg == "functions" and isinstance(kw.value, ast.List | ast.Tuple):
            return [e.value for e in kw.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if kw.arg == "functions" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return [kw.value.value]
    return []


def check_source(source: str, policy: dict | None = None) -> dict:
    """Run every static rule over ``source``. Returns a JSON-serialisable dict."""
    policy = policy or {}
    severity = {**DEFAULT_SEVERITY, **(policy.get("static") or {})}
    allowed_backends = set(policy.get("allowed_backends", ["cuda", "triton"]))
    env_allowlist = set(policy.get("env_allowlist", ["TORCH_CUDA_ARCH_LIST", "MAX_JOBS"]))
    findings: list[Finding] = []

    def add(rule: str, message: str, line: int | None = None) -> None:
        sev = severity.get(rule, "warn")
        if sev != "off":
            findings.append(Finding(rule, sev, message, line))

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        add("syntax_error", f"cannot parse candidate: {exc.msg}", exc.lineno)
        return _result(findings, backends=[], declared_kernels=[])

    col = _Collector()
    col.visit(tree)
    call_names = [(n, ln) for n, ln, _ in col.calls]
    string_blob = "\n".join(s for s, _ in col.strings)

    # --- structure -------------------------------------------------------------------------
    modelnew = [c for c in col.classes if c.name == "ModelNew"]
    if not modelnew:
        add("no_modelnew", "no class named ModelNew")
    for cls in modelnew:
        for base in cls.bases:
            bname = col.dotted(base) or ""
            if bname.split(".")[-1] == "Model":
                add("subclasses_reference", "ModelNew inherits from the reference Model", cls.lineno)
    for name, ln in call_names:
        if name == "Model" or name.endswith(".Model"):
            add("instantiates_reference", "constructs the reference Model", ln)

    # --- backends --------------------------------------------------------------------------
    backends: set[str] = set()
    load_inline_calls = [(n, ln, c) for n, ln, c in col.calls
                         if n.endswith("load_inline") or n.endswith("cpp_extension.load")]
    if load_inline_calls:
        backends.add("cuda")
    triton_kernels = []
    for fn in col.functions:
        for dec in fn.decorator_list:
            dname = col.dotted(dec.func if isinstance(dec, ast.Call) else dec) or ""
            if dname in {"triton.jit", "triton.autotune", "triton.heuristics"} or dname.endswith("triton.jit"):
                backends.add("triton")
                triton_kernels.append(fn.name)
    if any(n.startswith("cupy.") and n.endswith(("RawKernel", "RawModule", "ElementwiseKernel")) for n, _ in call_names):
        backends.add("cupy")
    if re.search(r"@T\.prim_func|tilelang", source):
        backends.add("tilelang")
    if re.search(r"\bcute::|\bcutlass::", string_blob) or any(i.startswith("cutlass") for i, _ in col.imports):
        backends.add("cutlass")
    if not backends:
        add("no_kernel_backend", "no custom kernel found (load_inline / cpp_extension / @triton.jit)")
    for b in sorted(backends - allowed_backends - {"cutlass"}):
        add("backend_not_allowed", f"backend '{b}' is not in policy.allowed_backends")

    # Exported load_inline functions must be called somewhere outside the load_inline call.
    used = {name.split(".")[-1] for name, _ in call_names}
    used |= {attr.split(".")[-1] for attr, _ in col.attributes}
    for name, _, call in col.calls:  # getattr(mod, "fn") also counts as a use
        if name == "getattr" and len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
            used.add(str(call.args[1].value))
    for _, ln, call in load_inline_calls:
        exports = _load_inline_exports(call)
        if exports:
            if not used.intersection(exports):
                add("kernel_defined_not_called",
                    f"load_inline exports {exports} but none of them is ever called", ln)
    for kname in triton_kernels:
        launched = any(t.split(".")[-1] == kname for t, _ in col.subscript_calls)
        referenced = any(name.split(".")[-1] == kname for name, _ in call_names) or \
            any(a.split(".")[-1] == kname for a, _ in col.attributes)
        if not launched and not referenced:
            add("triton_kernel_not_launched", f"@triton.jit kernel '{kname}' is never launched")

    # --- banned shortcuts ------------------------------------------------------------------
    for name, ln in call_names:
        if name in {"torch.compile", "torch.jit.script", "torch.jit.trace", "torch.export.export"} or \
                name.startswith(("torch._dynamo", "torch._inductor")):
            add("torch_compile", f"uses {name}", ln)
    for name, ln, _ in col.decorators:
        if name.startswith(("torch.compile", "torch.jit", "torch._dynamo")):
            add("torch_compile", f"decorated with {name}", ln)
    for mod, ln in col.imports:
        if mod.startswith(("torch._dynamo", "torch._inductor")):
            add("torch_compile", f"imports {mod}", ln)

    # --- tampering with numerics, environment or the harness --------------------------------
    for target, ln in col.assign_targets:
        if target.startswith("torch.backends."):
            add("numeric_flag_tamper", f"assigns {target}", ln)
        elif target.startswith(_HARNESS_ROOTS) and not target.startswith("self."):
            add("harness_tamper", f"assigns {target}", ln)
    numeric_calls = {"torch.set_float32_matmul_precision", "torch.set_default_dtype",
                     "torch.set_flush_denormal", "torch.backends.cudnn.flags",
                     "torch.backends.cuda.preferred_blas_library", "torch.set_default_device"}
    for name, ln in call_names:
        if name in numeric_calls:
            add("numeric_flag_tamper", f"calls {name}", ln)
        if name in {"setattr", "builtins.setattr"}:
            call = next(c for n, l2, c in col.calls if n == name and l2 == ln)
            if call.args:
                tgt = col.dotted(call.args[0]) or ""
                if tgt.startswith(("torch", "time", "sys", "builtins", "numpy", "os")):
                    add("harness_tamper", f"setattr on {tgt}", ln)
        if name in {"os.putenv", "os.environ.update", "os.environ.setdefault", "os.environ.__setitem__"}:
            add("env_tamper", f"calls {name}", ln)
    for base, key, ln in col.subscript_assigns:
        if base == "os.environ" and key not in env_allowlist:
            add("env_tamper", f"sets os.environ[{key!r}] (allowed: {sorted(env_allowlist)})", ln)
        if base == "sys.modules":
            add("harness_tamper", "writes to sys.modules", ln)

    for name, ln in call_names:
        if _matches(name, _INTROSPECTION):
            add("introspection", f"calls {name}", ln)
        if name in {"exec", "eval", "compile", "__import__", "importlib.import_module",
                    "importlib.util.spec_from_file_location", "runpy.run_path", "runpy.run_module"}:
            add("dynamic_code", f"calls {name}", ln)
        if _matches(name, _NETWORK_OR_PROCESS):
            add("process_or_network", f"calls {name}", ln)
        if name in {"open", "builtins.open", "io.open", "pathlib.Path.read_text", "numpy.load", "torch.load",
                    "pickle.load", "pickle.loads"} or name.endswith((".read_text", ".read_bytes")):
            add("file_io", f"calls {name} (reading cached results from disk is not allowed)", ln)
    for mod, ln in col.imports:
        root = mod.split(".")[0]
        if root in {"subprocess", "socket", "urllib", "requests", "http", "httpx", "aiohttp", "ftplib"}:
            add("process_or_network", f"imports {mod}", ln)
        if root in _THREAD_MODULES:
            add("threads", f"imports {mod}", ln)
        if root in {"inspect", "gc"}:
            add("introspection", f"imports {mod}", ln)
        if root in {"ctypes", "cffi"}:
            add("harness_tamper", f"imports {mod} (raw memory access)", ln)

    # --- lazy tensors / subclasses ---------------------------------------------------------
    for cls in col.classes:
        for base in cls.bases:
            bname = col.dotted(base) or ""
            if bname in {"torch.Tensor", "Tensor", "torch._C.TensorBase"}:
                add("lazy_tensor", f"class {cls.name} subclasses torch.Tensor", cls.lineno)
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name in {"__torch_function__", "__torch_dispatch__"}:
                add("lazy_tensor", f"defines {item.name}", item.lineno)
    for name, ln in call_names:
        if name.endswith("_make_subclass") or name.endswith("_make_wrapper_subclass"):
            add("lazy_tensor", f"calls {name}", ln)

    if col.tries:
        add("try_except", "try/except can hide a fallback to PyTorch when the kernel fails", col.tries[0])

    # --- streams, caching, precision -------------------------------------------------------
    for name, ln in call_names:
        if name in {"torch.cuda.Stream", "torch.cuda.stream", "torch.cuda.ExternalStream"} or \
                name.endswith((".wait_stream", ".record_stream")):
            add("side_stream", f"uses {name}; timing must include all streams", ln)
    for pat in _CUDA_STREAM_PATTERNS:
        if re.search(pat, string_blob):
            add("side_stream", f"CUDA source uses {pat}")
    for name, ln, _ in col.decorators:
        if name in {"functools.lru_cache", "functools.cache", "lru_cache", "cache", "functools.cached_property"}:
            add("memoisation", f"decorator {name} can cache outputs", ln)
    for name, ln in call_names:
        if name.endswith(".data_ptr") or name == "id":
            add("memoisation", f"calls {name}; keying on tensor identity can return stale results", ln)
            break

    torch_ops = sorted({n for n, _ in col.calls_in_modelnew
                        if _matches(n, _TORCH_COMPUTE_PREFIXES)
                        or (n.startswith("torch.nn.") and n.split(".")[2] not in _TORCH_NN_ALLOWED)})
    if torch_ops:
        add("torch_ops", "ModelNew still calls PyTorch compute ops: " + ", ".join(torch_ops))
    precision_hits = [p for p in _REDUCED_PRECISION_PATTERNS if re.search(p, source)]
    if any(n.endswith((".half", ".bfloat16")) for n, _ in call_names) or \
            any(a in {"torch.float16", "torch.bfloat16", "torch.half"} for a, _ in col.attributes):
        precision_hits.append("torch reduced-precision dtype")
    if precision_hits:
        add("reduced_precision", "uses reduced precision: " + ", ".join(precision_hits))

    # --- KernelBench's own checker, if installed --------------------------------------------
    kb = _run_kernelbench_checker(source, backends)
    for msg in kb:
        add("kernelbench_checker", msg)

    declared = sorted(set(triton_kernels) | set(_GLOBAL_RE.findall(string_blob)))
    return _result(findings, backends=sorted(backends), declared_kernels=declared)


# __global__ functions declared in CUDA source strings (template and __launch_bounds__ forms too).
_GLOBAL_RE = re.compile(r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(?:static\s+)?(?:inline\s+)?"
                        r"void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_]\w*)")


def _run_kernelbench_checker(source: str, backends: set[str]) -> list[str]:
    try:
        from kernelbench.kernel_static_checker import validate_kernel_static  # type: ignore
    except Exception:
        return []
    backend = "triton" if backends == {"triton"} else "cuda"
    try:
        _, errors, warnings = validate_kernel_static(source, backend=backend, precision="fp32")
    except Exception as exc:  # their checker evolves; never let it crash ours
        return [f"KernelBench checker failed to run: {exc}"]
    return [f"[kb error] {e}" for e in errors] + [f"[kb warning] {w}" for w in warnings]


def _result(findings: list[Finding], backends: list[str], declared_kernels: list[str]) -> dict:
    by = {"error": [], "warn": [], "info": []}
    for f in findings:
        by.setdefault(f.severity, []).append(asdict(f))
    return {
        "ok": not by["error"],
        "errors": by["error"],
        "warnings": by["warn"],
        "info": by["info"],
        "backends": backends,
        "declared_kernels": declared_kernels,
        "rules_fired": sorted({f.rule for f in findings}),
    }
