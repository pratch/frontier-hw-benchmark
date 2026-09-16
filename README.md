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

## Benchmark Evaluation Results

### Comparison Against Frontier Models (Overall Relaxed Accuracy)

| Model | Model Scale | Overall Relaxed Accuracy | Official Citation |
| :--- | :---: | :---: | :--- |
| **Qwen2-VL-2B-Instruct** | **2.2B** (Local, fp16) | **75.76%** | **This Work** (1x V100 profile) |
| Claude 3 Opus | Frontier Closed | 80.8% | Anthropic (2024) |
| Claude 3 Sonnet | Frontier Closed | 81.1% | Anthropic (2024) |
| GPT-4o | Frontier Closed | 85.7% | OpenAI (2024) |
| Claude 3.5 Sonnet | Frontier Closed | 90.8% | Anthropic (2024) |

### Detailed Performance Breakdown for Qwen2-VL-2B-Instruct

| Evaluation Split / Metric | Accuracy / Value | Samples | Description |
| :--- | :---: | :---: | :--- |
| **Overall Relaxed Accuracy** | **75.76%** | 2,500 | Full ChartQA test set |
| --- **ChartQA-H (Human Reasoning)** | **59.12%** | 1,250 | Multi-step reasoning, arithmetic $\Delta$, trends |
| --- **ChartQA-M (Machine Extractive)** | **92.40%** | 1,250 | Direct value extraction and OCR lookup |
| **Prompt Rewording Spread** | **$\Delta = 6.96\%$** | 2,500 $\times$ 2 | Spread across `direct` (75.76%) and `concise` (68.80%) |

### Prompt Templates Used (Prompt Rewording Spread)

To test prompt sensitivity and compute the spread interval, two meaning-preserving prompt templates are used:

- **Direct Prompt (`--template direct`)**:
  ```text
  Answer the question directly using the chart. Give a short, precise answer with no extra explanation. Question: {question}
  ```
- **Concise Prompt (`--template concise`)**:
  ```text
  Look at the chart and provide only the final answer to the question: {question}
  ```

---

## Summary of Findings (Presentation Takeaways)

1. **Overall Performance**: `Qwen2-VL-2B-Instruct` scores **75.76%** overall relaxed accuracy on the 2,500 ChartQA test questions, showing a noticeable gap to frontier closed models (80.8% – 90.8%).
2. **Sub-Split Asymmetry**: Extractive machine questions (`ChartQA-M`: **92.40%**) are straightforward and near saturation. The capability gap is concentrated in human reasoning questions (`ChartQA-H`: **59.12%**), which require multi-step mental calculation (averaging across bars, calculating ratios, comparing series sums) that fail under greedy single-step decoding.
3. **Prompt Sensitivity as a Confound**: Rewording the prompt from `direct` to `concise` drops accuracy by **6.96%** (and by **11.92%** on the human reasoning split), demonstrating that prompt styling alone is a major hidden confound when benchmarking VLMs.
4. **Presentation Deck**: Slide deck for a 4–5 min project presentation is available in [`chartqa-benchmark.pptx`](chartqa-benchmark.pptx). Selected sample questions and failure cases are documented in [`data_samples/`](data_samples/).

---

## Defense Claim

> *"Model Qwen/Qwen2-VL-2B-Instruct at revision 'main', fp16 on GPU 1 (RTX 3090 Ti / V100 profile), greedy decoding, template 'direct', scored 75.76% overall relaxed accuracy (range [68.80%, 75.76%] over meaning-preserving prompt rewordings); the literature reports 80.8% for Claude 3 Opus, 81.1% for Claude 3 Sonnet, 85.7% for GPT-4o, and 90.8% for Claude 3.5 Sonnet. The difference is confounded with model scale (2.2B vs >1T), visual resolution encoder capacity, pretraining data mixture, and chain-of-thought reasoning prompting."*

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
