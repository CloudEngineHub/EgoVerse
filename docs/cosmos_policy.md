# Cosmos Policy Environment Installation

This guide describes how to set up the emimic environment with Cosmos Policy integration for the EgoVerse project.

## Prerequisites

- Python 3.10 (required for cu128 extra compatibility)
- [uv](https://github.com/astral-sh/uv) package manager
- NVIDIA GPU with CUDA 12.8 support


## Installation

For convenience, here's the complete sequence of commands:

```bash
cd /coc/flash7/bli678/Projects/EgoVerse
uv venv emimic --python 3.10
uv pip install --python emimic/bin/python -r requirements.txt
uv pip install --python emimic/bin/python -e external/lerobot
uv pip install --python emimic/bin/python -e .
# Install cosmos-policy with cu128 extra
cd external/cosmos-policy
uv pip install --python ../../emimic/bin/python -e ".[cu128]" \
  --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu128_torch27/simple
cd ../..
# Install LIBERO group dependencies
uv pip install --python emimic/bin/python \
  bddl cloudpickle draccus easydict gym "imageio[ffmpeg]" libero "mujoco==3.3.2"
```


