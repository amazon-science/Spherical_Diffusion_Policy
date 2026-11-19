"""SO(3) equivariance tests for the spherical-Fourier core of Spherical Diffusion Policy.

What this guards
----------------
The paper's central claim is that the denoiser is SE(3) equivariant: the network
applied to a rotated input equals the correspondingly-rotated output of the
unrotated input, ``f(rho(R) . x) == rho(R) . f(x)``.  T(3) invariance is handled
outside these modules (the policy re-centres the point cloud), so what is left
to check numerically is the SO(3) half, degree by degree.

The modules below are the atoms that ``IrrepConditionalUnet1D`` -- the
"Spherical Denoising Temporal Unet" of the paper -- is built out of:

  * ``IrrepConv1dBlock``  - the spherical temporal convolution and, with
    ``kernel_size=1``, the SFiLM conditioner itself
    (sdp/model/equivariant_diffusion/irreps_conditional_unet1d.py:54)
  * ``IrrepLinear``       - the SSFiLM conditioner
  * ``EquivariantRMSNormArraySphericalHarmonicsV2`` - reached via ``norm=True``
  * ``SeparableS2Activation`` + ``SO3_Grid`` - reached via ``activation=True``
  * ``SO3_LinearV2`` / ``SO3_Embedding`` - the equiformer trunk's linear map

``IrrepConditionalUnet1D`` itself is deliberately *not* imported here; see the
note at the bottom of this file.

Representation convention
-------------------------
Feature tensors are ``(B, C * (lmax + 1) ** 2, N)`` with the irrep index as the
*minor* axis inside the channel axis -- that is the layout
``IrrepConv1dBlock.forward`` both consumes and produces (it reads
``'b (c i) n -> b c i n'`` and writes ``'(b n) i c -> b (c i) n'``).

The group action ``rho(R)`` is NOT hand-rolled: it comes from the repo's own
``SO3_Rotation``, which builds the block-diagonal Wigner-D matrix from a 3x3
rotation using its vendored copy of e3nn 0.4.0's ``wigner_D``
(sdp/model/equiformer_v2/wigner.py:16).  The test therefore tracks the code's own
conventions, including the vendored ``Jd.pt`` basis.

Why these assertions can fail
-----------------------------
The equivariant ops share one scalar weight across all ``2l + 1`` components of
a degree ``l`` and add bias to ``l = 0`` only.  Any edit that breaks that breaks
these tests.  Measured in float64 on a faithful re-implementation of the same
index algebra, the real op lands at ``2e-16`` while

  * swapping two m-components inside the l=1 block gives ... 1.9e+00
  * replacing the op with a plain dense ``nn.Conv1d`` gives . 1.3e+00
  * adding the l=0 bias to every degree gives .............. 2.9e-01

``EXACT_TOL = 1e-4`` therefore sits ~100x above float32 round-off and >1000x
below the weakest of those breaks.  The first two are asserted below as negative
controls, so a tolerance loosened until it passes anything would itself turn the
suite red.
"""

import sys
from pathlib import Path

import einops
import pytest
import torch
import torch.nn as nn


def _repo_root():
    """Locate the checkout root (the directory containing ``sdp/``).

    ``setup.py`` uses ``find_packages()`` and no directory under ``sdp/`` has an
    ``__init__.py``, so ``pip install -e .`` installs nothing importable.  Put
    the checkout on ``sys.path`` explicitly and rely on PEP-420 namespace
    packages.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sdp" / "model" / "equiformer_v2" / "so3.py").is_file():
            return parent
    raise RuntimeError(f"could not locate the sdp/ package above {here}")


sys.path.insert(0, str(_repo_root()))

# Hard imports, not importorskip: if e3nn moves out from under this repo the
# suite must go red rather than silently skip.
from e3nn import o3  # noqa: E402

from sdp.model.equiformer_v2.module_list import ModuleListInfo  # noqa: E402
from sdp.model.equiformer_v2.so3 import (  # noqa: E402
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from sdp.model.equivariant_diffusion.irreps_conv1d_components import (  # noqa: E402
    IrrepConv1dBlock,
    IrrepLinear,
)

torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

LMAX = 2                      # sdp/config/sdp_ddpm_5layer.yaml sets lmax: 2
D_IRREP = (LMAX + 1) ** 2     # 9 spherical-harmonic coefficients
GRID_RESOLUTION = 14          # IrrepConditionalUnet1D default grid_resolution
SEED = 0

B, N = 2, 8                   # batch, trajectory length
C_IN, C_OUT = 3, 5

# Everything runs in float32: SO3_Rotation.RotationToWignerDMatrix allocates its
# Wigner buffer with ``torch.zeros(..., device=...)`` and no dtype
# (sdp/model/equiformer_v2/so3.py:452), so a float64 input would hit a dtype
# mismatch in the torch.bmm that follows.  float32 puts the achievable floor at
# ~1e-6 relative through one block.
DTYPE = torch.float32
EXACT_TOL = 1e-4

# The S2 pointwise activation is only equivariant up to the aliasing of the
# spherical quadrature: SiLU of a band-limited signal is not band-limited, so
# ``from_grid(SiLU(to_grid(.)))`` folds l > lmax energy back into l <= lmax by an
# orientation-dependent amount.  That floor is a property of the method, not a
# bug, so the activated path is scored by a ratio against a matched broken
# control as well as by an absolute budget.
ALIASING_TOL = 5e-2
CONTROL_MARGIN = 20.0

# Fixed rotations, built with the same e3nn helper the repo uses to go from
# Euler angles to a matrix (so3.py:445).  beta is kept away from 0 and pi, where
# the acos inside o3.xyz_to_angles loses precision.
_EULER_ANGLES = [
    (0.37, 0.81, 1.23),
    (1.57, 2.11, -0.64),
    (-2.25, 1.34, 0.92),
]


@pytest.fixture(params=_EULER_ANGLES, ids=["rotA", "rotB", "rotC"])
def rotation(request):
    alpha, beta, gamma = request.param
    return o3.angles_to_matrix(
        torch.tensor(alpha, dtype=DTYPE),
        torch.tensor(beta, dtype=DTYPE),
        torch.tensor(gamma, dtype=DTYPE),
    )


@pytest.fixture(scope="module")
def so3_grid():
    """The nested (lmax, mmax) grid table, built exactly as the SDTU builds it.

    Mirrors sdp/model/equivariant_diffusion/irreps_conditional_unet1d.py:146-158.
    """
    table = ModuleListInfo("({}, {})".format(LMAX, LMAX))
    for degree in range(LMAX + 1):
        row = nn.ModuleList()
        for order in range(LMAX + 1):
            row.append(
                SO3_Grid(
                    degree,
                    order,
                    resolution=GRID_RESOLUTION,
                    normalization="component",
                )
            )
        table.append(row)
    return table


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wigner(rot_3x3, batch):
    """The repo's own rho(R), replicated ``batch`` times."""
    rot = SO3_Rotation(LMAX)
    rot.set_wigner(rot_3x3.unsqueeze(0).repeat(batch, 1, 1).contiguous())
    return rot


def _rho_feature(rot, x):
    """Apply rho(R) to a ``(B, C * D_IRREP, N)`` tensor via ``SO3_Rotation``."""
    _, _, n = x.shape
    emb = einops.rearrange(x, "b (c i) n -> (b n) i c", i=D_IRREP)
    emb = rot.rotate(emb, LMAX, LMAX)
    return einops.rearrange(emb, "(b n) i c -> b (c i) n", n=n)


def _rel_err(lhs, rhs):
    scale = rhs.abs().max().clamp(min=1e-12)
    return ((lhs - rhs).abs().max() / scale).item()


def _equivariance_error(fn, x, rot):
    """``|| f(rho x) - rho f(x) ||_inf / || f(x) ||_inf``"""
    with torch.no_grad():
        return _rel_err(fn(_rho_feature(rot, x)), _rho_feature(rot, fn(x)))


def _input(c_in=C_IN, n=N, std=1.0, seed=SEED + 100):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, c_in * D_IRREP, n, generator=gen, dtype=DTYPE) * std


def _make_block(so3_grid, kernel_size=3, norm=False, activation=False):
    torch.manual_seed(SEED)
    block = IrrepConv1dBlock(
        C_IN,
        C_OUT,
        kernel_size,
        so3_grid,
        norm=norm,
        max_lmax=LMAX,
        activation=activation,
    )
    block.eval()
    assert block.d_irrep == D_IRREP
    return block


def _jiggle(param, seed, scale=0.3):
    """Move a parameter off its constant initialisation so the test exercises it."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        param.add_(scale * torch.randn(param.shape, generator=gen, dtype=param.dtype))


class _SwapTwoOrdersOfDegreeOne(nn.Module):
    """A deliberately broken wrapper: equivariant block, then swap two
    m-components of the l=1 block on the output.  Degree-preserving but
    order-mixing, which is exactly the class of index bug an equivariance test
    exists to catch -- every shape, dtype and parameter is unchanged."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        y = self.block(x)
        y = einops.rearrange(y, "b (c i) n -> b c i n", i=D_IRREP).clone()
        y[:, :, [1, 2], :] = y[:, :, [2, 1], :]
        return einops.rearrange(y, "b c i n -> b (c i) n")


# ---------------------------------------------------------------------------
# the group action itself
# ---------------------------------------------------------------------------


def test_so3_rotation_yields_a_nontrivial_orthogonal_representation(rotation):
    """Guard against a vacuous suite: rho(R) must be a real, non-identity isometry.

    If rho(R) were the identity every equivariance assertion below would hold
    trivially, so this is checked before anything is asserted about it.
    """
    rot = _wigner(rotation, batch=4)
    wigner = rot.wigner

    assert wigner.shape == (4, D_IRREP, D_IRREP)
    assert torch.isfinite(wigner).all()

    eye = torch.eye(D_IRREP, dtype=wigner.dtype)
    gram = wigner[0] @ wigner[0].transpose(-1, -2)
    assert _rel_err(gram, eye) < 1e-4, "Wigner-D is not orthogonal"
    assert (wigner[0] - eye).abs().max() > 0.1, "rho(R) is (near) the identity"

    # block diagonal: degree l must not leak into degree l'
    offsets = [0, 1, 4, 9]
    for a in range(LMAX + 1):
        for b in range(LMAX + 1):
            if a == b:
                continue
            block = wigner[0, offsets[a]:offsets[a + 1], offsets[b]:offsets[b + 1]]
            assert block.abs().max() < 1e-5, f"degree {a} leaks into degree {b}"

    # l = 0 is the trivial irrep
    assert abs(wigner[0, 0, 0].item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# exactly equivariant ops
# ---------------------------------------------------------------------------


def test_irrep_conv1d_block_is_equivariant(rotation, so3_grid):
    """The bare spherical temporal convolution: one scalar weight per degree,
    bias on l = 0 only.  Exactly equivariant, up to float32."""
    block = _make_block(so3_grid, kernel_size=3)
    err = _equivariance_error(block, _input(), _wigner(rotation, B * N))
    assert err < EXACT_TOL, f"IrrepConv1dBlock equivariance error {err:.3e}"


def test_irrep_conv1d_block_kernel5_is_equivariant(rotation, so3_grid):
    """kernel_size=5 is what sdp_ddpm_5layer.yaml actually trains with."""
    block = _make_block(so3_grid, kernel_size=5)
    err = _equivariance_error(block, _input(), _wigner(rotation, B * N))
    assert err < EXACT_TOL, f"kernel_size=5 equivariance error {err:.3e}"


def test_irrep_conv1d_block_with_rms_norm_sh_is_equivariant(rotation, so3_grid):
    """``norm=True`` routes through EquivariantRMSNormArraySphericalHarmonicsV2.

    Its scale is built from per-degree norms weighted uniformly inside each
    degree (layer_norm.py:329-336), making it a rotation *invariant*, so the
    composition stays exactly equivariant.  A per-m weight there would break it.
    """
    block = _make_block(so3_grid, kernel_size=3, norm=True)
    _jiggle(block.norm.affine_weight, SEED + 7)
    _jiggle(block.norm.affine_bias, SEED + 8)
    err = _equivariance_error(block, _input(), _wigner(rotation, B * N))
    assert err < EXACT_TOL, f"rms_norm_sh equivariance error {err:.3e}"


def test_sfilm_conditioner_is_equivariant(rotation, so3_grid):
    """The SFiLM cond_encoder, constructed with the SDTU's own arguments
    (irreps_conditional_unet1d.py:54-55): kernel_size=1, norm off, activation
    off, ``2 * out_channels`` outputs.  It sees a single conditioning token, so
    N = 1."""
    torch.manual_seed(SEED)
    cond_encoder = IrrepConv1dBlock(
        C_IN, 2 * C_OUT, 1, so3_grid, norm=False, max_lmax=LMAX, activation=False
    )
    cond_encoder.eval()
    err = _equivariance_error(cond_encoder, _input(n=1), _wigner(rotation, B))
    assert err < EXACT_TOL, f"SFiLM cond_encoder equivariance error {err:.3e}"


def test_irrep_linear_is_equivariant(rotation):
    """``IrrepLinear`` is the SSFiLM conditioner; its einops patterns only admit
    N = 1."""
    torch.manual_seed(SEED)
    layer = IrrepLinear(C_IN, 2 * C_OUT, LMAX)
    layer.eval()
    _jiggle(layer.bias, SEED + 11, scale=0.5)  # off its zero init
    err = _equivariance_error(layer, _input(n=1), _wigner(rotation, B))
    assert err < EXACT_TOL, f"IrrepLinear equivariance error {err:.3e}"


def test_so3_linear_v2_is_equivariant(rotation):
    """``SO3_LinearV2`` over ``SO3_Embedding`` -- the equiformer trunk's linear map.

    Scored natively in the ``(n_nodes, coefficients, channels)`` layout rather
    than the conv layout, so this also pins down ``SO3_Embedding``'s contract.
    """
    n_nodes = 6
    torch.manual_seed(SEED)
    layer = SO3_LinearV2(C_IN, C_OUT, LMAX)
    layer.eval()
    _jiggle(layer.bias, SEED + 13, scale=0.5)

    gen = torch.Generator().manual_seed(SEED + 17)
    raw = torch.randn(n_nodes, D_IRREP, C_IN, generator=gen, dtype=DTYPE)
    rot = _wigner(rotation, n_nodes)

    def run(coeffs):
        emb = SO3_Embedding(
            0, [LMAX], C_IN, device=coeffs.device, dtype=coeffs.dtype
        )
        emb.set_embedding(coeffs)
        emb.set_lmax_mmax([LMAX], [LMAX])
        return layer(emb).embedding

    with torch.no_grad():
        lhs = run(rot.rotate(raw, LMAX, LMAX))
        rhs = rot.rotate(run(raw), LMAX, LMAX)
    err = _rel_err(lhs, rhs)
    assert err < EXACT_TOL, f"SO3_LinearV2 equivariance error {err:.3e}"


# ---------------------------------------------------------------------------
# negative controls: these prove the assertions above are not vacuous
# ---------------------------------------------------------------------------


def test_plain_conv1d_is_not_equivariant(rotation):
    """A standard dense Conv1d over the same flattened (channel, irrep) axis --
    what you get if the irrep structure is dropped -- must be rejected by the
    same metric, with orders of magnitude to spare."""
    torch.manual_seed(SEED)
    plain = nn.Conv1d(C_IN * D_IRREP, C_OUT * D_IRREP, 3, padding=1)
    plain.eval()
    err = _equivariance_error(plain, _input(), _wigner(rotation, B * N))
    assert err > 100 * EXACT_TOL, (
        f"dense Conv1d control looks equivariant (err {err:.3e}); the "
        "equivariance metric or the group action is broken"
    )


def test_order_mixing_inside_a_degree_is_not_equivariant(rotation, so3_grid):
    """Swapping two m-components of l=1 keeps every shape, dtype, degree count
    and parameter identical, and still must be caught."""
    broken = _SwapTwoOrdersOfDegreeOne(
        _make_block(so3_grid, kernel_size=3, norm=True)
    ).eval()
    err = _equivariance_error(broken, _input(), _wigner(rotation, B * N))
    assert err > 100 * EXACT_TOL, (
        f"order-mixing control looks equivariant (err {err:.3e})"
    )


# ---------------------------------------------------------------------------
# the S2 activation path (approximately equivariant by construction)
# ---------------------------------------------------------------------------


def test_s2_activation_path_is_equivariant_within_quadrature_aliasing(
    rotation, so3_grid
):
    """Full block: irrep conv -> rms_norm_sh -> SeparableS2Activation.

    This is the only path that touches e3nn's ``ToS2Grid`` / ``FromS2Grid``, and
    the one an e3nn bump can silently break: the Wigner-D here comes from the
    repo's *vendored* e3nn-0.4.0 ``Jd.pt`` while the grid matrices come from the
    *installed* e3nn, so the two must keep agreeing on the real
    spherical-harmonic basis.  If they ever stop, this is where it shows up.

    Scored both absolutely and as a ratio against a matched broken control, so
    a pass cannot be bought by loosening a constant.
    """
    block = _make_block(so3_grid, kernel_size=3, norm=True, activation=True)
    x = _input(std=0.5)
    rot = _wigner(rotation, B * N)

    err = _equivariance_error(block, x, rot)
    control = _equivariance_error(_SwapTwoOrdersOfDegreeOne(block).eval(), x, rot)

    assert err < ALIASING_TOL, (
        f"activated block equivariance error {err:.3e} exceeds the "
        f"quadrature-aliasing budget {ALIASING_TOL:.1e}"
    )
    assert err * CONTROL_MARGIN < control, (
        f"activated block error {err:.3e} is not clearly separated from the "
        f"broken control {control:.3e}"
    )


def test_s2_activation_keeps_degree_zero_invariant(rotation, so3_grid):
    """``SeparableS2Activation`` routes l = 0 through a plain SiLU of the gating
    scalars, so the l = 0 output of the activated block must be *exactly*
    rotation invariant -- a far tighter statement than the aliasing-limited
    bound above, and one that fails at once if the scalar and tensor branches
    get crossed."""
    block = _make_block(so3_grid, kernel_size=3, norm=True, activation=True)
    x = _input(std=0.5)
    rot = _wigner(rotation, B * N)

    with torch.no_grad():
        y_plain = block(x)
        y_rot = block(_rho_feature(rot, x))

    def degree_zero(y):
        return einops.rearrange(y, "b (c i) n -> b c i n", i=D_IRREP)[:, :, 0, :]

    err = _rel_err(degree_zero(y_rot), degree_zero(y_plain))
    assert err < EXACT_TOL, f"degree-0 output is not invariant (err {err:.3e})"


# ---------------------------------------------------------------------------
# NOT COVERED, on purpose
# ---------------------------------------------------------------------------
# ``IrrepConditionalUnet1D`` -- the assembled SDTU, and what the paper's claim is
# really about -- is not exercised here because
# sdp/model/equivariant_diffusion/irreps_conditional_unet1d.py cannot be
# imported on a CPU-only runner.  All three blockers are dead weight:
#
#   line  9  import pytorch3d.transforms as pytorch3d_transforms   <- never used
#   line 13  from ...equiformerv2_block import FeedForwardNetwork   <- never used
#                                                   (pulls in torch_geometric)
#   line 14  from ...se3_transformation import rot_pcd              <- used only
#            inside the `if __name__ == "__main__":` demo at lines 299-364
#            (se3_transformation.py imports pytorch3d)
#
# Deleting those three lines, and the demo's three ``rot_pcd`` calls, would let
# the whole SDTU be tested end to end with nothing but torch, einops and e3nn.
# That is a source change, so it belongs in its own PR rather than smuggled in
# behind a test.
