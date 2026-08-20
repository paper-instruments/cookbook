# Environment Setup

## Install

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training

# Option A: conda
conda create -n cookbook python=3.12 -y && conda activate cookbook
python -m pip install -e .

# Option B: uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

The cookbook pins the exact SDK source required by its recipes in
`pyproject.toml`, alongside recipe-only dependencies such as `tinker-cookbook`.

## Credentials

Set your API key via `.env` (auto-loaded by `python-dotenv`) or environment variable:

```bash
# Option A: .env file in training/
echo 'FIREWORKS_API_KEY="your-api-key"' > .env

# Option B: export
export FIREWORKS_API_KEY="your-api-key"
```

## Verify

```bash
python -c "import fireworks.training.sdk; print('SDK OK')"
python -c "import training.recipes.rl_loop, training.recipes.dpo_loop; print('Recipes OK')"
```

The recipe import check is intentional: a clean base install should run the
standard DPO/RL recipes without optional example-only packages such as
`eval-protocol`. Install the `dev` extra only when running tests or
eval-protocol examples.

## Dev dependencies (tests, coverage)

```bash
uv pip install -e ".[dev]"   # or: python -m pip install -e ".[dev]"
python -m pytest tests/
```

## Upgrading the SDK

The required SDK source is pinned in `pyproject.toml`. To upgrade:

```bash
cd cookbook/training
uv pip install --upgrade -e .
```

Then verify the installed source matches the pin:

```bash
grep 'fireworks-ai\[training\]' pyproject.toml
python tests/verify_sdk_minimum.py --assert-installed
```
