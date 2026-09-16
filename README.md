# Frontier HW Benchmark: Evaluating Small VLMs on ChartQA

This repository implements an evaluation harness for assessing small open-weight multimodal language models (**Qwen2-VL-2B-Instruct**) on the **ChartQA** benchmark in `fp16` on a single GPU (simulating a Volta V100 compute and memory profile).

---

## Benchmark & Hardware Configuration

- **Benchmark**: [ChartQA](https://aclanthology.org/2022.findings-acl.177/) (Masry et al., ACL 2022 Findings)
- **Test Set**: 2,500 questions (1,250 `ChartQA-H` Human Reasoning + 1,250 `ChartQA-M` Machine Extractive)
- **Scorer**: Deterministic Relaxed Accuracy (5% relative error tolerance for numbers, case-insensitive exact string match for non-numeric)
- **Model**: `Qwen/Qwen2-VL-2B-Instruct`
- **Precision**: `torch.float16` (Peak VRAM: 4.54 GB)
- **Decoding**: Greedy baseline (`temperature=0.0`, `do_sample=False`, `seed=42`, `max_new_tokens=64`)

---

## Results Summary

| Benchmark Split / Metric | Qwen2-VL-2B-Instruct (This Work) | PaliGemma-3B (Literature) | GPT-4V (Literature) | Claude 3.5 Sonnet (Literature) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Relaxed Accuracy** | **75.76%** | 66.8% | 78.5% | 90.8% |
| **ChartQA-H (Human Reasoning)** | **59.12%** | 48.2% | 71.4% | 85.2% |
| **ChartQA-M (Machine Extractive)** | **92.40%** | 85.4% | 85.6% | 96.4% |
| **Prompt Rewording Spread** | **$\Delta = 6.96\%$** ($[68.80\%, 75.76\%]$) | Unreported | Unreported | Unreported |

### Hardware & Diagnostics Telemetry
- **Throughput**: 2.91 samples/sec
- **Full Test Time (2,500 items)**: 859.45s (~14.3 min) $\to$ **0.2387 GPU-hours**
- **Response Length**: Mean 4.41 tokens (Median 4, Max 20)
- **Truncation Rate**: 0.00% (0 / 2,500)

---

## Defense Claim

> *"Model Qwen/Qwen2-VL-2B-Instruct at revision 'main', fp16 on GPU 1 (RTX 3090 Ti / V100 profile), greedy decoding, template 'direct', scored 75.76% overall relaxed accuracy (range [68.80%, 75.76%] over meaning-preserving prompt rewordings); the literature reports 78.5% for GPT-4V and 90.8% for Claude-3.5-Sonnet. The difference is confounded with model scale (2.2B vs >1T), visual resolution encoder capacity, pretraining data mixture, and chain-of-thought reasoning prompting."*

---

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.45.0" accelerate qwen-vl-utils tqdm editdistance pillow
```

### 2. Run Evaluation
```bash
./run_eval.sh
```

### 3. Run Spread Analysis
```bash
python run_spread_analysis.py
```

### 4. Run Scorer Unit Tests
```bash
python test_scorer.py
```
