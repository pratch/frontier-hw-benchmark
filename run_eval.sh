#!/bin/bash
set -euo pipefail

# ==============================================================================
# One-command Entry Point for ChartQA Evaluation
# Class Project: One Term, One V100 (simulated on GPU 1 with fp16)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Target GPU 1 (RTX 3090 Ti)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

# Python Virtual Environment
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

echo "======================================================================"
echo "Starting ChartQA Evaluation on GPU 1 (RTX 3090 Ti / V100 profile)"
echo "Model: Qwen/Qwen2-VL-2B-Instruct | Precision: float16 | Seed: 42"
echo "======================================================================"

"$VENV_DIR/bin/python" -u "$SCRIPT_DIR/evaluate_chartqa.py" \
    --model_id "Qwen/Qwen2-VL-2B-Instruct" \
    --data_dir "$SCRIPT_DIR/data/ChartQA Dataset" \
    --output_dir "$SCRIPT_DIR/results" \
    --gpu_id 0 \
    --template "direct" \
    --seed 42 \
    --max_new_tokens 64 \
    "$@"
