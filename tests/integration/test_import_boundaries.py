"""Static enforcement of the package import boundaries (task 4.2).

The authoritative rules live in ``DESIGN`` section 3.2 ("Import rules") and
``docs/DECISIONS.md`` AD-27 (the single evaluator-only import allowlist); the
layer/dependency order is drawn in ``docs/ARCHITECTURE.md`` section 4.1. This
module reads every ``*.py`` file under ``src/cyberswarm`` and checks those rules
*statically*, by parsing the source with :mod:`ast`. It never imports the
modules under test, so it needs none of the optional dependencies (``torch``,
``gymnasium`` ...) and cannot be fooled by prose in docstrings or comments.

Rules enforced (each as its own test):

* **Package dependency order.** A package may import, at module load, only the
  sibling ``cyberswarm`` packages and the named third-party libraries listed for
  it in DESIGN 3.2 (plus itself, the standard library and ``pydantic`` - the
  always-installed contract-record foundation that every ``ContractModel`` and
  ``common.ConfigModel`` is built on). Lazy imports inside function bodies are
  exempt, which is how ``common.seeding`` and ``graph.tensors`` may reach
  ``torch`` without making it a hard dependency.
* **Evaluator-only allowlist (AD-27).** Only ``interfaces.simulator``,
  ``interfaces.ports``, ``rl.reward``, ``engine.step_engine`` and the ``mock``,
  ``evaluation`` and ``runner`` packages may import
  ``cyberswarm.contracts.evaluator_only`` / ``GroundTruthSnapshot``. (The
  ``contracts`` package itself defines it and is not restricted.)
* **Ground-truth call sites.** Only ``engine.step_engine`` (and the test suite)
  may *call* ``ground_truth()`` / ``termination_reason()``. The reusable
  ``interfaces.conformance`` mixin is test support and is exempt.
* **``ValidatedAction`` construction.** Only ``coordinator.validator``
  constructs a :class:`ValidatedAction`. The ``interfaces.conformance`` mixin is
  the one exempt test-support site (DESIGN 3.2: "the mock may construct it only
  in tests").
* **The ``mappo`` token.** No file under ``src`` contains the token ``mappo``
  (case-insensitive) except ``contracts/manifest.py`` and ``runner/export.py``.

The allowlists cover the whole planned package set, but only packages that
actually exist on disk are scanned, so the suite stays green while later stages
(``mock``, ``graph``, ``rl`` ...) are still unwritten.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Locate the package tree (repo-relative, no reliance on the install layout).
# --------------------------------------------------------------------------- #
def _find_package_root() -> Path:
    """Return ``.../src/cyberswarm`` by walking up from this test file."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "cyberswarm"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"could not locate src/cyberswarm starting from {here}")


PKG_ROOT = _find_package_root()
SRC_ROOT = PKG_ROOT.parent  # .../src
REPO_ROOT = SRC_ROOT.parent  # repository root

# --------------------------------------------------------------------------- #
# Classification constants.
# --------------------------------------------------------------------------- #
#: Standard-library top-level module names (plus ``__future__``) are always ok.
_STDLIB: frozenset[str] = frozenset(sys.stdlib_module_names) | {"__future__"}

#: pydantic is the shared contract-modelling foundation (every record is a
#: pydantic model; ``common.ConfigModel`` subclasses ``pydantic.BaseModel``). It
#: is a core, always-installed dependency, so it is permitted everywhere, like
#: the standard library, and is never treated as a boundary-crossing import.
_FOUNDATIONAL: frozenset[str] = frozenset({"pydantic"})

#: Every lead-owned package name under ``cyberswarm`` (DESIGN 3.2 table).
_ALL_PACKAGES: frozenset[str] = frozenset(
    {
        "contracts",
        "common",
        "interfaces",
        "mock",
        "roles",
        "graph",
        "coordinator",
        "evaluation",
        "engine",
        "rl",
        "baselines",
        "marl",
        "uncertainty",
        "runner",
    }
)

#: Allowed module-load imports per package: the sibling ``cyberswarm`` packages
#: and the named third-party libraries (DESIGN 3.2). ``torch`` is intentionally
#: omitted for ``common``, ``graph`` and ``uncertainty`` because those reach it
#: only lazily (``common.seeding``, ``graph.tensors.to_torch``) or only through
#: ``marl``; a module-level torch import there is a violation.
ALLOWED_IMPORTS: dict[str, frozenset[str]] = {
    "contracts": frozenset(),
    "common": frozenset({"contracts", "numpy"}),
    "interfaces": frozenset({"contracts", "common", "numpy"}),
    "mock": frozenset({"contracts", "interfaces", "common", "numpy", "networkx"}),
    "roles": frozenset({"contracts", "interfaces", "common"}),
    "graph": frozenset({"contracts", "common", "numpy", "networkx"}),
    "coordinator": frozenset({"contracts", "interfaces", "common"}),
    "evaluation": frozenset({"contracts", "common", "numpy"}),
    "engine": frozenset({"contracts", "interfaces", "common", "roles", "graph"}),
    "rl": frozenset(
        {
            "contracts",
            "interfaces",
            "common",
            "roles",
            "graph",
            "engine",
            "numpy",
            "gymnasium",
            "pettingzoo",
        }
    ),
    "baselines": frozenset({"contracts", "interfaces", "common", "roles"}),
    "marl": frozenset(
        {"contracts", "interfaces", "common", "rl", "engine", "numpy", "torch"}
    ),
    "uncertainty": frozenset({"contracts", "interfaces", "common", "marl", "numpy"}),
    # The runner wires everything together; it may import every lead-owned
    # package and any of the named third-party libraries.
    "runner": _ALL_PACKAGES
    | frozenset({"numpy", "networkx", "torch", "gymnasium", "pettingzoo"}),
}

#: The one evaluator-only import allowlist (DESIGN 3.2 / AD-27).
_EVAL_ONLY_EXACT: frozenset[str] = frozenset(
    {
        "cyberswarm.interfaces.simulator",
        "cyberswarm.interfaces.ports",
        "cyberswarm.rl.reward",
        "cyberswarm.engine.step_engine",
    }
)
_EVAL_ONLY_PACKAGES: tuple[str, ...] = (
    "cyberswarm.mock",
    "cyberswarm.evaluation",
    "cyberswarm.runner",
)
#: The evaluator-only record family is anchored by ``GroundTruthSnapshot``;
#: importing that name (from anywhere) pulls in hidden truth.
_EVAL_ONLY_NAMES: frozenset[str] = frozenset({"GroundTruthSnapshot"})
_EVAL_ONLY_MODULE_SUFFIX = "contracts.evaluator_only"

#: Only ``engine.step_engine`` (and tests) may call these; the conformance mixin
#: is test support and is exempt.
_GT_METHODS: frozenset[str] = frozenset({"ground_truth", "termination_reason"})
_GT_CALL_ALLOWED: frozenset[str] = frozenset({"cyberswarm.engine.step_engine"})
_GT_CALL_EXEMPT: frozenset[str] = frozenset(
    {
        "cyberswarm.interfaces.conformance",  # test-support mixin
        # The mock is the lead-owned, clearly-labelled GroundTruthAccessor
        # *provider*: its _MockGroundTruthAccessor delegates to the simulator's
        # own ground_truth()/termination_reason(). AD-27 restricts decision-time
        # *readers* of ground truth, not the provider, so the mock is exempt.
        "cyberswarm.mock.simulator",
    }
)

#: Only ``coordinator.validator`` constructs ``ValidatedAction``; the conformance
#: mixin is the one exempt test-support site.
_VALIDATED_ACTION = "ValidatedAction"
_VA_ALLOWED: frozenset[str] = frozenset({"cyberswarm.coordinator.validator"})
_VA_EXEMPT: frozenset[str] = frozenset({"cyberswarm.interfaces.conformance"})

#: The only files that may contain the ``mappo`` token (repo-relative, posix).
_MAPPO_ALLOWED: frozenset[str] = frozenset(
    {
        "src/cyberswarm/contracts/manifest.py",
        "src/cyberswarm/runner/export.py",
    }
)


# --------------------------------------------------------------------------- #
# Parse every source module once.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ModuleInfo:
    """A parsed source file and the identity needed to apply the rules."""

    path: Path
    rel: str  # posix path relative to the repo root
    module: str  # dotted module name, e.g. "cyberswarm.interfaces.simulator"
    package: str  # top-level subpackage, e.g. "interfaces" ("" for the root)
    file_package: str  # dotted path of the containing directory (its __package__)
    source: str
    tree: ast.Module


def _load_modules() -> list[_ModuleInfo]:
    modules: list[_ModuleInfo] = []
    for path in sorted(PKG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel_src = path.relative_to(SRC_ROOT)
        stem_parts = rel_src.with_suffix("").parts
        module_parts = stem_parts[:-1] if stem_parts[-1] == "__init__" else stem_parts
        module = ".".join(module_parts)
        dir_parts = rel_src.parts[:-1]
        file_package = ".".join(dir_parts)
        package = dir_parts[1] if len(dir_parts) >= 2 else ""
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        modules.append(
            _ModuleInfo(
                path=path,
                rel=path.relative_to(REPO_ROOT).as_posix(),
                module=module,
                package=package,
                file_package=file_package,
                source=source,
                tree=tree,
            )
        )
    return modules


_MODULES: list[_ModuleInfo] = _load_modules()


# --------------------------------------------------------------------------- #
# AST helpers.
# --------------------------------------------------------------------------- #
def _is_type_checking_test(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` guards."""

    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


class _ModuleLevelImports(ast.NodeVisitor):
    """Collect import nodes that execute at module load.

    Imports inside a function body are skipped (they are lazy, the sanctioned
    escape hatch for optional deps such as ``torch`` in ``to_torch``), and so
    are imports guarded by ``if TYPE_CHECKING:`` - those run only under a static
    type checker, never at runtime, so they create no runtime dependency.
    """

    def __init__(self) -> None:
        self.nodes: list[ast.Import | ast.ImportFrom] = []
        self._fn_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn_depth += 1
        self.generic_visit(node)
        self._fn_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            # The body is type-checking only; only the runtime else-branch counts.
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self._fn_depth == 0:
            self.nodes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._fn_depth == 0:
            self.nodes.append(node)


def _resolve_relative(file_package: str, level: int, module: str | None) -> str:
    """Resolve a relative import to its absolute dotted module name."""

    parts = file_package.split(".") if file_package else []
    if level - 1 > 0:
        parts = parts[: max(0, len(parts) - (level - 1))]
    base = ".".join(parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def _subpackage(dotted: str) -> str | None:
    """Return the ``cyberswarm`` subpackage of a dotted module, or ``None``."""

    parts = dotted.split(".")
    if not parts or parts[0] != "cyberswarm":
        return None
    return parts[1] if len(parts) >= 2 else None


def _dependency_tokens(info: _ModuleInfo) -> set[str]:
    """Return the set of boundary-crossing dependencies loaded at import time.

    A token is either a sibling ``cyberswarm`` package name or a third-party
    top-level module name. The standard library, ``pydantic`` and self-imports
    are excluded.
    """

    visitor = _ModuleLevelImports()
    visitor.visit(info.tree)
    tokens: set[str] = set()

    for node in visitor.nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "cyberswarm":
                    sub = _subpackage(alias.name)
                    if sub and sub != info.package:
                        tokens.add(sub)
                elif root in _STDLIB or root in _FOUNDATIONAL:
                    continue
                elif root:
                    tokens.add(root)
            continue

        # ImportFrom
        if node.level:
            base = _resolve_relative(info.file_package, node.level, node.module)
            if node.module is None:
                targets = [f"{base}.{alias.name}" for alias in node.names]
            else:
                targets = [base]
            for target in targets:
                sub = _subpackage(target)
                if sub and sub != info.package:
                    tokens.add(sub)
            continue

        module = node.module or ""
        root = module.split(".")[0]
        if root == "cyberswarm":
            if module == "cyberswarm":
                targets = [f"cyberswarm.{alias.name}" for alias in node.names]
            else:
                targets = [module]
            for target in targets:
                sub = _subpackage(target)
                if sub and sub != info.package:
                    tokens.add(sub)
        elif root in _STDLIB or root in _FOUNDATIONAL or root == "":
            continue
        else:
            tokens.add(root)

    return tokens


def _imports_evaluator_only(info: _ModuleInfo) -> bool:
    """True if the module imports the evaluator-only module or GroundTruthSnapshot."""

    for node in ast.walk(info.tree):
        if isinstance(node, ast.Import):
            if any(a.name.endswith(_EVAL_ONLY_MODULE_SUFFIX) for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                target = _resolve_relative(info.file_package, node.level, node.module)
            else:
                target = node.module or ""
            if target.endswith(_EVAL_ONLY_MODULE_SUFFIX):
                return True
            if any(a.name in _EVAL_ONLY_NAMES for a in node.names):
                return True
    return False


def _called_method_names(info: _ModuleInfo) -> set[str]:
    """Return the set of attribute/name call targets in the module."""

    names: set[str] = set()
    for node in ast.walk(info.tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def _calls_ground_truth(info: _ModuleInfo) -> bool:
    return bool(_called_method_names(info) & _GT_METHODS)


def _constructs_validated_action(info: _ModuleInfo) -> bool:
    return _VALIDATED_ACTION in _called_method_names(info)


def _eval_only_allowed(module: str) -> bool:
    if module in _EVAL_ONLY_EXACT:
        return True
    return any(
        module == pkg or module.startswith(pkg + ".") for pkg in _EVAL_ONLY_PACKAGES
    )


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
def test_source_tree_is_scannable() -> None:
    """The scanner actually found the currently-implemented packages."""

    packages = {m.package for m in _MODULES if m.package}
    assert {"contracts", "common", "interfaces"} <= packages, packages
    # A path mistake would make every other check pass vacuously; guard it.
    assert len(_MODULES) >= 10, f"only found {len(_MODULES)} modules under {PKG_ROOT}"


def test_package_dependency_order() -> None:
    """Every module imports only what DESIGN 3.2 permits for its package."""

    offenders: list[str] = []
    for info in _MODULES:
        allowed = ALLOWED_IMPORTS.get(info.package)
        if allowed is None:
            # Root package file, or a future package not yet in the table.
            continue
        for token in sorted(_dependency_tokens(info)):
            if token not in allowed:
                offenders.append(
                    f"{info.rel}: package '{info.package}' may not import '{token}'"
                )
    assert not offenders, "forbidden module-level imports:\n" + "\n".join(offenders)


def test_evaluator_only_import_allowlist() -> None:
    """Only the AD-27 allowlist imports evaluator-only ground truth."""

    offenders: list[str] = []
    importers: list[str] = []
    for info in _MODULES:
        if info.package == "contracts":
            continue  # contracts defines evaluator_only; it is not restricted
        if _imports_evaluator_only(info):
            importers.append(info.module)
            if not _eval_only_allowed(info.module):
                offenders.append(info.rel)
    assert not offenders, (
        "modules importing evaluator-only truth outside the allowlist:\n"
        + "\n".join(offenders)
    )
    # Positive control: the detector really fires on a known allowed importer.
    assert "cyberswarm.interfaces.simulator" in importers


def test_ground_truth_calls_restricted() -> None:
    """Only engine.step_engine (and tests) call ground_truth/termination_reason."""

    offenders: list[str] = []
    callers: list[str] = []
    for info in _MODULES:
        if not _calls_ground_truth(info):
            continue
        callers.append(info.module)
        if info.module in _GT_CALL_EXEMPT:
            continue  # conformance mixin: test support
        if info.module not in _GT_CALL_ALLOWED:
            offenders.append(info.rel)
    assert not offenders, (
        "ground_truth()/termination_reason() called outside engine.step_engine:\n"
        + "\n".join(offenders)
    )
    # Positive control: the call detector fires on the conformance mixin.
    assert "cyberswarm.interfaces.conformance" in callers


def test_validated_action_construction_restricted() -> None:
    """Only coordinator.validator constructs ValidatedAction (conformance exempt)."""

    offenders: list[str] = []
    builders: list[str] = []
    for info in _MODULES:
        if not _constructs_validated_action(info):
            continue
        builders.append(info.module)
        if info.module in _VA_EXEMPT:
            continue  # conformance mixin: documented test-support constructor
        if info.module not in _VA_ALLOWED:
            offenders.append(info.rel)
    assert not offenders, (
        "ValidatedAction constructed outside coordinator.validator:\n"
        + "\n".join(offenders)
    )
    # Positive control: the construction detector fires on the conformance mixin.
    assert "cyberswarm.interfaces.conformance" in builders


def test_mappo_token_confined() -> None:
    """The 'mappo' token appears only in the two files DESIGN 3.2 names."""

    offenders: list[str] = []
    for info in _MODULES:
        if "mappo" in info.source.lower() and info.rel not in _MAPPO_ALLOWED:
            offenders.append(info.rel)
    assert not offenders, "the 'mappo' token leaked into:\n" + "\n".join(offenders)
