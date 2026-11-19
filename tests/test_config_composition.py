"""Composition and resolution checks for the hydra configs under ``sdp/config/``.

What this guards
----------------
``train.py`` is driven entirely by hydra: it declares
``@hydra.main(version_base=None, config_path=.../sdp/config)`` with no
``config_name``, so every run names its config on the command line. A broken
defaults list, a dangling interpolation, or a ``_target_`` pointing at a module
that no longer exists are all invisible until someone launches a training job.
These tests make them visible in CI instead.

Deliberately NOT instantiated
-----------------------------
``train.py`` does ``hydra.utils.get_class(cfg._target_)``, and essentially every
policy / dataset / env_runner target imports something that cannot be installed
on a CPU-only runner (pytorch3d, dgl, torch_geometric, robomimic, gym, av,
open3d). So these tests compose and fully resolve the configs, then verify each
``_target_`` *statically* -- by checking the module file exists on disk -- rather
than importing it. That keeps the check runnable while still catching a rename.

Requires only hydra-core (which brings omegaconf) and pytest.
"""

import ast
import operator
import pathlib

import pytest
from omegaconf import OmegaConf

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "sdp" / "config"
TASK_DIR = CONFIG_DIR / "task"
TRAIN_PY = REPO_ROOT / "train.py"

TOP_LEVEL = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
TASK_FRAGMENTS = sorted(p.stem for p in TASK_DIR.glob("*.yaml"))

# Counts are asserted below so the gate cannot silently shrink to nothing if a
# glob or a directory layout changes.
EXPECTED_TOP_LEVEL = 14
EXPECTED_TASK_FRAGMENTS = 6


# ---------------------------------------------------------------------------
# a restricted arithmetic evaluator, standing in for train.py's ${eval:}
# ---------------------------------------------------------------------------
# train.py:59 registers Python's builtin ``eval`` as an OmegaConf resolver. The
# configs use it 59 times, and every expression in the tree is plain arithmetic
# over interpolated numbers -- the six distinct shapes are all of the form
# ``round(50000 / ${n_demo})`` or ``${n_obs_steps} - 1``.
#
# This test suite does NOT register builtin ``eval``. Evaluating arbitrary
# strings is unnecessary here, and this repository's remediation path is a
# security control, so test code should not hand a general evaluator a config
# value. The whitelist below covers exactly what the configs need and raises on
# anything else, which also means a config that starts relying on a richer
# expression fails loudly here rather than silently diverging from production.

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_CALLS = {"round": round, "int": int, "float": float, "min": min, "max": max}


def safe_arithmetic_eval(expression):
    """Evaluate a numeric expression without using builtin ``eval``.

    Supports number literals, + - * / // % **, unary +/-, parentheses, and the
    calls in ``_ALLOWED_CALLS``. Anything else raises ValueError.
    """

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError(f"only numeric literals allowed, got {node.value!r}")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_walk(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _ALLOWED_CALLS
            and not node.keywords
        ):
            return _ALLOWED_CALLS[node.func.id](*[_walk(a) for a in node.args])
        raise ValueError(
            f"disallowed expression element {type(node).__name__} in {expression!r}"
        )

    return _walk(ast.parse(str(expression).strip(), mode="eval"))


def _max_steps_from_train_py():
    """Read train.py's module-level ``max_steps`` dict without importing it.

    train.py imports sdp.workspace.base_workspace and rebinds sys.stdout at
    module scope, so it must not be imported in a test process. Reading the
    literal with ast keeps this in step with train.py automatically.
    """
    for node in ast.parse(TRAIN_PY.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "max_steps" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict) and value, "max_steps is empty"
            return value
    raise AssertionError(f"no module-level `max_steps = {{...}}` found in {TRAIN_PY}")


def _register_resolvers():
    """Register the four resolvers train.py registers (train.py:44-59)."""
    max_steps = _max_steps_from_train_py()
    OmegaConf.register_new_resolver(
        "get_max_steps", lambda name: max_steps[name], replace=True
    )
    OmegaConf.register_new_resolver(
        "get_ws_x_center",
        lambda name: -0.2
        if name.startswith("kitchen_") or name.startswith("hammer_cleanup_")
        else 0.0,
        replace=True,
    )
    OmegaConf.register_new_resolver("get_ws_y_center", lambda _name: 0.0, replace=True)
    OmegaConf.register_new_resolver("eval", safe_arithmetic_eval, replace=True)

    # ${now:...} appears 104 times and is normally registered by hydra itself.
    # If a future hydra moves setup_globals, fall back to a stand-in: a hydra
    # internal rename must not turn every config test red.
    try:
        from hydra.core.utils import setup_globals

        setup_globals()
    except Exception:  # pragma: no cover
        from datetime import datetime

        OmegaConf.register_new_resolver(
            "now",
            lambda pattern="%Y-%m-%d %H:%M:%S": datetime.now().strftime(pattern),
            use_cache=True,
            replace=True,
        )


_register_resolvers()


def _compose(config_name, overrides=()):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def _targets(container):
    """Every ``_target_`` string anywhere in a resolved config."""
    found = []
    if isinstance(container, dict):
        for key, value in container.items():
            if key == "_target_" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_targets(value))
    elif isinstance(container, list):
        for item in container:
            found.extend(_targets(item))
    return found


def _module_file_exists(dotted_target):
    """True if ``a.b.C`` corresponds to a module file in this checkout.

    The last path component is a class or function, so it is dropped. Both
    ``a/b.py`` and ``a/b/__init__.py`` count.
    """
    parts = dotted_target.split(".")
    for cut in (len(parts) - 1, len(parts)):
        module_parts = parts[:cut]
        if not module_parts:
            continue
        base = REPO_ROOT.joinpath(*module_parts)
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_config_inventory_has_not_shrunk():
    """A gate that silently stops covering anything is worse than no gate."""
    assert CONFIG_DIR.is_dir(), f"{CONFIG_DIR} is missing"
    assert len(TOP_LEVEL) == EXPECTED_TOP_LEVEL, (
        f"expected {EXPECTED_TOP_LEVEL} top-level configs, found "
        f"{len(TOP_LEVEL)}: {TOP_LEVEL}. If this change is intended, update "
        "EXPECTED_TOP_LEVEL."
    )
    assert len(TASK_FRAGMENTS) == EXPECTED_TASK_FRAGMENTS, (
        f"expected {EXPECTED_TASK_FRAGMENTS} task fragments, found "
        f"{len(TASK_FRAGMENTS)}: {TASK_FRAGMENTS}. If intended, update "
        "EXPECTED_TASK_FRAGMENTS."
    )


# ---------------------------------------------------------------------------
# the restricted evaluator itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("round(50000 / 100)", 500),
        ("round(1000 / 10)", 100),
        ("50000 / 1000", 50.0),
        ("8 - 1", 7),
        ("-3 + 10", 7),
        ("2 * (3 + 4)", 14),
    ],
)
def test_safe_arithmetic_eval_computes_config_expressions(expression, expected):
    assert safe_arithmetic_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd').read()",
        "lambda: 1",
        "[].__class__",
        "'a' * 3",
    ],
)
def test_safe_arithmetic_eval_refuses_non_arithmetic(expression):
    """The whole point of not registering builtin ``eval``."""
    with pytest.raises((ValueError, SyntaxError)):
        safe_arithmetic_eval(expression)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", TOP_LEVEL)
def test_top_level_config_composes_and_fully_resolves(config_name):
    """Compose the config as train.py would, then force every interpolation.

    ``throw_on_missing=True`` turns a dangling ``???`` into a failure, and
    ``resolve=True`` turns an unresolvable ``${...}`` into one.
    """
    cfg = _compose(config_name)
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(container, dict) and container
    assert "_target_" in container, f"{config_name} declares no top-level _target_"


@pytest.mark.parametrize("fragment", TASK_FRAGMENTS)
def test_task_fragment_composes_under_some_parent(fragment):
    """Every task fragment must compose under at least one top-level config.

    Two of the six fragments are orphans -- no defaults list names them -- so
    they are exercised here via a ``task=<fragment>`` override. A fragment
    interpolates keys its parent defines, so it is tried against each parent and
    need only succeed under one; if none works the fragment is unreachable and
    that is a real finding, not a skip.
    """
    failures = []
    for parent in TOP_LEVEL:
        try:
            cfg = _compose(parent, overrides=[f"task={fragment}"])
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            return
        except Exception as exc:  # noqa: BLE001 - we report the aggregate below
            failures.append(f"{parent}: {type(exc).__name__}: {str(exc)[:120]}")
    pytest.fail(
        f"task fragment {fragment!r} composed under none of the "
        f"{len(TOP_LEVEL)} top-level configs:\n  " + "\n  ".join(failures)
    )


@pytest.mark.parametrize("config_name", TOP_LEVEL)
def test_every_target_points_at_a_module_in_this_checkout(config_name):
    """Catch a renamed or deleted module without importing it.

    Only in-repo ``sdp.*`` targets are checked; third-party targets (e.g. a
    torch optimiser) are not this repository's to guarantee.
    """
    cfg = _compose(config_name)
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    targets = _targets(container)
    assert targets, f"{config_name} resolved to no _target_ at all"

    broken = [
        t for t in targets if t.startswith("sdp.") and not _module_file_exists(t)
    ]
    assert not broken, (
        f"{config_name} references targets whose module file does not exist "
        f"in this checkout: {sorted(set(broken))}"
    )
