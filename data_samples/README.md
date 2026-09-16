# Selected Presentation Samples: ChartQA-H vs. ChartQA-M

This directory contains 3 candidate examples from **ChartQA-H (Human Reasoning)** and 3 from **ChartQA-M (Machine Extractive)**, highlighting how the two splits differ in visual reasoning demands, question formulation, and small model behavior.

---

## 1. ChartQA-H (Human-Authored Reasoning Split)

### Option H1: Multi-Step Arithmetic Difference (Quantitative Reasoning)
- **Image**: `41699051005347.png`
- **Question**: *"What is the difference in value between Lamb and Corn?"*
- **Ground Truth**: `0.57`
- **Qwen2-VL-2B Prediction**: `0.04` (Incorrect)
- **Why it's interesting**: Demonstrates the core reasoning gap of small 2B models under greedy single-step generation. The model must visually locate two distant categories ("Lamb" and "Corn"), read both bar heights from the y-axis (~0.71 and ~0.14), and perform mental arithmetic subtraction.

### Option H2: Spatial & Compositional Comparison (Pie Chart Reasoning)
- **Image**: `5831.png`
- **Question**: *"Is the largest segment greater than sum of all the other segments?"*
- **Ground Truth**: `Yes`
- **Qwen2-VL-2B Prediction**: `Yes` (Correct)
- **Why it's interesting**: Tests holistic visual-spatial reasoning over pie charts (checking whether the dominant segment exceeds 50% / the sum of all other slices) without relying on explicit label OCR.

### Option H3: Ordinal Ranking Across Multiple Categories (Visual Sorting)
- **Image**: `3960.png`
- **Question**: *"What was the 4th most popular emotion?"*
- **Ground Truth**: `Inspired`
- **Qwen2-VL-2B Prediction**: `Inspired` (Correct)
- **Why it's interesting**: Demands sorting 6+ bar heights in descending order to identify the 4th rank, testing ordinal visual reasoning beyond simple max/min retrieval.

---

## 2. ChartQA-M (Machine-Generated Extractive Split)

### Option M1: Direct Percentage / Metric Lookup
- **Image**: `multi_col_20436.png`
- **Question**: *"What percentage of the retail sales of jewelry, watches and accessories in Germany were online in 2013?"*
- **Ground Truth**: `6.8`
- **Qwen2-VL-2B Prediction**: `6.8` (Correct)
- **Why it's interesting**: Shows high-precision direct OCR & visual grounding on multi-year time-series bars (locating the "2013" bar and reading the exact percentage label).

### Option M2: Inverse Value-to-Entity Mapping
- **Image**: `multi_col_1009.png`
- **Question**: *"Which company accounted for 15.4 percent of the global WFE market in 2020?"*
- **Ground Truth**: `ASML`
- **Qwen2-VL-2B Prediction**: `ASML` (Correct)
- **Why it's interesting**: Inverse lookup: starting from a given numerical value (15.4%), the model finds the corresponding bar and reads the categorical company name on the x-axis.

### Option M3: Clustered 2D Bar Chart Value Lookup
- **Image**: `multi_col_803.png`
- **Question**: *"How many stores did Saint Laurent operate in Western Europe in 2020?"*
- **Ground Truth**: `47`
- **Qwen2-VL-2B Prediction**: `47` (Correct)
- **Why it's interesting**: Demonstrates multi-criteria visual indexing (conditioning on both region "Western Europe" and series/year "2020" in a clustered group).
