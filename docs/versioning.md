# Versioning & Release Basics

This is a minimal guide for building, tagging, and installing this Python package.

## 1. Build wheel and source archive

```bash
pdm install
pdm build
```

Output will be saved under `dist/`, for example:
- `dist/core_llm_bridge-0.1.0-py3-none-any.whl`
- `dist/core_llm_bridge-0.1.0.tar.gz`

## 2. Create a git tag

```bash
git add .
git commit -m "Release v0.1.0"
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0
```

## 3. Install from GitHub with pdm

```bash
pdm add git+https://github.com/fermaat/core-llm-bridge.git@v0.1.0
```

or install the latest main branch:

```bash
pdm add git+https://github.com/fermaat/core-llm-bridge.git@main
```

## 4. Install locally from the built wheel

```bash
pip install dist/core_llm_bridge-0.1.0-py3-none-any.whl
```

## 5. Bump version

- Update `version` in `pyproject.toml`
- Re-build with `pdm build`
- Create a new tag `v0.1.1`, push it, and install from that tag
