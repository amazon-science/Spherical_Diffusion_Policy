# Security Policy

## Reporting a vulnerability

If you discover a potential security issue in this project, please notify AWS/Amazon Security via the
[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/).

Please do **not** create a public GitHub issue for a security report.

This matches the guidance in [CONTRIBUTING.md](CONTRIBUTING.md); the vulnerability reporting page is
the authoritative channel.

## Supported versions

This repository is the reference implementation accompanying the ICML 2025 paper *Spherical Diffusion
Policy: A SE(3) Equivariant Visuomotor Policy with Spherical Fourier Representation*. The code on
`main` is the supported version. There are no released or separately maintained branches.

## Dependency updates

Dependency security fixes are automated:

- `requirements.txt` is the pip manifest used for GitHub dependency-graph coverage, which is what
  allows Dependabot alerts and automated security fix pull requests to be raised at all.
- `conda_environment.yaml` remains the authoritative environment for reproducing the paper's results.
- Dependabot pull requests are gated by the CI workflow in `.github/workflows/ci.yml`. A pull request
  that fails CI is not merged.

## Known limitations of automated dependency scanning

Stated explicitly so that an absence of alerts is not mistaken for an absence of risk:

- **Conda-only dependencies are not covered by alerts.** `pytorch3d`, `dgl`, `pytorch-cluster` and
  `pytorch-scatter` are installed from conda channels. GitHub's dependency graph does not parse conda
  environment files, so no alert can be raised for them.
- **Four further dependencies cannot be expressed in a pip manifest that installs.** `gym==0.21.0`
  and `av==10.0.0` publish source distributions only (gym's `setup.py` fails under modern pip; `av`
  needs FFmpeg development headers), and `open3d` requires system GL. They are listed in the header
  of `requirements.txt` rather than as requirements, so they are also outside alert coverage.
- **Three dependencies are pinned to GitHub archive tarballs** at fixed commits (`robosuite`,
  `robomimic`, `robosuite-task-zoo`). Dependabot cannot version-bump an archive URL.
- **Vendored third-party source is not a tracked dependency.** As recorded in [NOTICE](NOTICE), this
  project includes code copied from EquiformerV2, Equivariant Diffusion Policy, Diffusion Policy, ACT
  and MimicGen. Copied source is not a declared dependency and is not scanned by Dependabot.
- **The pinned interpreter is `python=3.9`**, which reached end of life in October 2025 and therefore
  no longer receives upstream security patches.

Reproducing the paper's reported results requires the exact pinned environment in
`conda_environment.yaml`. Upgrading dependencies may change numerical results.
