# also add env
source /capstor/store/cscs/swissai/a144/jiaqchen/egoverse/faive2lerobot/faive2lerobot/bin/activate
export PYTHONPATH=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse:$PYTHONPATH
export PYTHONPATH=/capstor/store/cscs/swissai/a144/jiaqchen/egoverse/EgoVerse/external/lerobot:$PYTHONPATH

export HF_HOME=/iopsstor/scratch/cscs/jiaqchen/.hf_cache
export HF_DATASETS_CACHE=$HF_HOME/hf_datasets_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers_cache
export HF_HUB_CACHE=$HF_HOME/hub_cache

echo "HF_HOME: $HF_HOME"
echo "DONE SOURCING"