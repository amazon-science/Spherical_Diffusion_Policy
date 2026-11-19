"""Import-smoke tests for the subset of ``sdp`` that is importable on a CPU runner.

Why this is the broadest check available
----------------------------------------
172 of the repository's Python modules exist; 115 of them import something that
cannot be installed on a GitHub-hosted CPU runner (pytorch3d, dgl,
torch_geometric, torch_cluster, robomimic, gym, av, open3d, numba at module
scope, ...). The 54 listed below are the ones that reach only the standard
library, ``sdp`` itself, and the dependency set in requirements.txt.

That makes this the test that actually exercises a dependency bump: if
Dependabot raises torch, einops, e3nn, escnn, zarr, hydra or scipy and the new
version changes an API these modules touch at import time, this goes red and the
auto-merge does not happen. Several of these modules do real work at import --
``sdp/model/equiformer_v2/wigner.py:8`` runs ``torch.load`` on a checked-in
``Jd.pt``, and five functions across ``drop.py`` and ``gaussian_rbf.py`` are
``@torch.jit.script``, which compiles on import -- so this is a genuine
compatibility check rather than a syntax check.

Three modules resolve statically but must never be imported in a test process;
they are listed in ``NEVER_IMPORT`` with the reason.
"""

import ast
import importlib
import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ``setup.py`` uses ``find_packages()`` and there is no ``__init__.py`` anywhere
# under ``sdp/``, so ``pip install -e .`` installs nothing importable. The
# checkout has to go on sys.path, and ``sdp`` resolves as a PEP-420 namespace
# package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Third-party top-level names requirements.txt is allowed to supply at module
# scope. Anything outside this set appearing in a module below means the
# importable subset has quietly grown a new dependency.
ALLOWED_THIRD_PARTY = frozenset(
    {
        "torch",
        "numpy",
        "scipy",
        "e3nn",
        "escnn",
        "hydra",
        "omegaconf",
        "einops",
        "zarr",
        "numcodecs",
        "dill",
        "termcolor",
        "tqdm",
    }
)

NEVER_IMPORT = {
    "setup": "setup.py:3 calls setuptools.setup() at module scope",
    "sdp.model.detr.setup": (
        "calls setup() at module scope and reads 'README.md' with a "
        "CWD-relative path, which raises FileNotFoundError from the repo root"
    ),
    "train": (
        "train.py:9-10 rebinds sys.stdout/sys.stderr via open(fileno()), which "
        "raises under pytest capture and corrupts the session's streams"
    ),
}

IMPORTABLE_MODULES = [
    "sdp.common.checkpoint_util",
    "sdp.common.nested_dict_util",
    "sdp.common.normalize_util",
    "sdp.common.pose_trajectory_interpolator",
    "sdp.common.precise_sleep",
    "sdp.common.pytorch_util",
    "sdp.common.replay_buffer",
    "sdp.common.timestamp_accumulator",
    "sdp.dataset.base_dataset",
    "sdp.env_runner.base_image_runner",
    "sdp.env_runner.base_lowdim_runner",
    "sdp.model.bet.action_ae.discretizers.k_means",
    "sdp.model.bet.libraries.loss_fn",
    "sdp.model.bet.libraries.mingpt",
    "sdp.model.bet.libraries.mingpt.model",
    "sdp.model.bet.libraries.mingpt.trainer",
    "sdp.model.bet.libraries.mingpt.utils",
    "sdp.model.common.dict_of_tensor_mixin",
    "sdp.model.common.module_attr_mixin",
    "sdp.model.common.normalizer",
    "sdp.model.common.shape_util",
    "sdp.model.common.tensor_util",
    "sdp.model.detr.util",
    "sdp.model.diffusion.conditional_unet1d",
    "sdp.model.diffusion.conv1d_components",
    "sdp.model.diffusion.dp3_conditional_unet1d",
    "sdp.model.diffusion.ema_model",
    "sdp.model.diffusion.mask_generator",
    "sdp.model.diffusion.positional_embedding",
    "sdp.model.diffusion.transformer_for_diffusion",
    "sdp.model.equi.equi_conditional_unet1d",
    "sdp.model.equi.equi_conditional_unet1d_2",
    "sdp.model.equi.equi_encoder",
    "sdp.model.equiformer_v2.activation",
    "sdp.model.equiformer_v2.drop",
    "sdp.model.equiformer_v2.edge_rot_mat",
    "sdp.model.equiformer_v2.gaussian_rbf",
    "sdp.model.equiformer_v2.layer_norm",
    "sdp.model.equiformer_v2.module_list",
    "sdp.model.equiformer_v2.radial_function",
    "sdp.model.equiformer_v2.so2_ops",
    "sdp.model.equiformer_v2.so3",
    "sdp.model.equiformer_v2.wigner",
    "sdp.model.equivariant_diffusion.irreps_conv1d_components",
    "sdp.model.equivariant_diffusion.vn_conditional_unet1d",
    "sdp.model.equivariant_diffusion.vn_conv1d_components",
    "sdp.model.equivariant_vision.vec_layers",
    "sdp.model.unet.obs_cond_unet1d",
    "sdp.model.vision.voxel_crop_randomizer",
    "sdp.policy.base_image_policy",
    "sdp.policy.base_lowdim_policy",
    "sdp.policy.ibc_dfo_lowdim_policy",
    "sdp.shared_memory.shared_ndarray",
    "sdp.workspace.base_workspace",
]

EXPECTED_IMPORTABLE_COUNT = 54


def _module_path(dotted):
    """Map a dotted module name to its file in this checkout."""
    parts = dotted.split(".")
    base = REPO_ROOT.joinpath(*parts)
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return base.with_suffix(".py")


def _module_scope_imports(path):
    """Top-level package names imported at MODULE scope only.

    Imports nested inside a function, class body, ``try/except ImportError`` or
    ``if TYPE_CHECKING`` do not block importing the module, so they are not
    counted. ``sdp.common.replay_buffer`` is in the list above precisely because
    its ``numba`` import is deferred into a function.
    """
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:  # module scope only, deliberately not ast.walk
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def _is_third_party(name):
    """True if ``name`` resolves to something in site-packages."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return True  # unresolvable at module scope is worse than third-party
    if spec is None or spec.origin in (None, "built-in", "frozen"):
        return False
    return "site-packages" in spec.origin


# ---------------------------------------------------------------------------


def test_importable_set_has_not_shrunk():
    """Stop the smoke gate from quietly covering less than it claims."""
    assert len(IMPORTABLE_MODULES) == EXPECTED_IMPORTABLE_COUNT
    assert len(set(IMPORTABLE_MODULES)) == len(IMPORTABLE_MODULES), "duplicate entry"


def test_the_sdp_namespace_package_resolves():
    """Guards the sys.path/PEP-420 arrangement the rest of the suite relies on."""
    import sdp

    assert pathlib.Path(sdp.__path__[0]).resolve() == (REPO_ROOT / "sdp").resolve()


@pytest.mark.parametrize("dotted", IMPORTABLE_MODULES)
def test_module_file_exists(dotted):
    path = _module_path(dotted)
    assert path.is_file(), f"{dotted} -> {path} is missing"


@pytest.mark.parametrize("dotted", IMPORTABLE_MODULES)
def test_module_imports(dotted):
    """The actual compatibility check against the installed dependency set."""
    importlib.import_module(dotted)


@pytest.mark.parametrize("dotted", IMPORTABLE_MODULES)
def test_module_gained_no_new_third_party_dependency(dotted):
    """Catch a heavy import being added at module scope to a light module.

    Without this, someone adding ``import open3d`` to one of these files would
    not break any test here -- the module would simply drop out of the
    importable set and the smoke coverage would shrink silently.
    """
    offenders = {
        name
        for name in _module_scope_imports(_module_path(dotted))
        if name != "sdp" and name not in ALLOWED_THIRD_PARTY and _is_third_party(name)
    }
    assert not offenders, (
        f"{dotted} imports {sorted(offenders)} at module scope, which is not in "
        f"requirements.txt. Either add the dependency and update "
        f"ALLOWED_THIRD_PARTY, or defer the import into a function."
    )


@pytest.mark.parametrize("dotted", sorted(NEVER_IMPORT))
def test_side_effect_modules_are_excluded_from_the_smoke_set(dotted):
    """Documents why three statically-importable modules are not smoke-tested."""
    assert dotted not in IMPORTABLE_MODULES, (
        f"{dotted} must not be in the smoke set: {NEVER_IMPORT[dotted]}"
    )
