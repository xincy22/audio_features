# Release Guide

This file is a lightweight checklist for publishing to PyPI.

## Pre-release

- Update version in `pyproject.toml`.
- Update `CHANGELOG.md`.
- Run tests:
  - `python -m unittest discover -s tests`
- Build locally:
  - `python -m pip install -U build twine`
  - `python -m build`
- Validate artifacts:
  - `python -m twine check dist/*`

## Upload to TestPyPI (recommended)

- Create a TestPyPI API token.
- Set credentials (PowerShell):
  - `$env:TWINE_USERNAME="__token__"`
  - `$env:TWINE_PASSWORD="pypi-..."`
- Upload:
  - `python -m twine upload --repository testpypi dist/*`

## Upload to PyPI

- Create a PyPI API token.
- Set credentials (PowerShell):
  - `$env:TWINE_USERNAME="__token__"`
  - `$env:TWINE_PASSWORD="pypi-..."`
- Upload:
  - `python -m twine upload dist/*`

## Post-release

- Tag the release in git.
- Push tags to the remote.
- Update documentation if needed.
