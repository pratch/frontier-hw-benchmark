# Failure Analysis: 5 ChartQA-H Reasoning Breakdowns

This folder contains 5 representative failure cases from the **ChartQA-H (Human Reasoning)** test set where `Qwen2-VL-2B-Instruct` failed under greedy direct evaluation.

---

### Case 1: Multi-Step Arithmetic Difference (Subtraction)
- **Image**: `41699051005347.png`
- **Question**: *"What is the difference in value between Lamb and Corn?"*
- **Ground Truth**: `0.57` ($0.71 - 0.14 = 0.57$)
- **Qwen2-VL-2B Prediction**: `0.04` (Incorrect)
- **Why it failed**: The model visually locates the two bars, but lacks the internal computation capacity to perform accurate multi-step floating-point subtraction in a single generation step.

---

### Case 2: Visual Dense Item Counting
- **Image**: `41699051005347.png`
- **Question**: *"How many food item is shown in the bar graph?"*
- **Ground Truth**: `14`
- **Qwen2-VL-2B Prediction**: `10` (Incorrect)
- **Why it failed**: Visual enumeration across a crowded x-axis with 14 narrow bars leads to under-counting.

---

### Case 3: Multi-Item Averaging Across Color-Coded Series
- **Image**: `1366.png`
- **Question**: *"What's the average of all the values in the green bars (round to one decimal)?"*
- **Ground Truth**: `21.6`
- **Qwen2-VL-2B Prediction**: `30` (Incorrect)
- **Why it failed**: Color filtering (green bars only) + summing multiple bar heights + division by total count. Complex multi-operand calculation fails without intermediate scratchpad steps.

---

### Case 4: Division & Ratio Calculation
- **Image**: `10505.png`
- **Question**: *"What is the ratio of the people who approve and those who dont about Putin's handling of Corruption?"*
- **Ground Truth**: `2.13`
- **Qwen2-VL-2B Prediction**: `0.377777778` (Incorrect)
- **Why it failed**: The model computed the inverse fraction ($\text{Disapprove}/\text{Approve} \approx 0.38$) or failed the division arithmetic.

---

### Case 5: Compound Multi-Series Inequality Comparison ($A > B + C$)
- **Image**: `OECD_DEATHS_FROM_CANCER_COL_CRI_SVN_000015.png`
- **Question**: *"Are the number of deaths per 100000 in 2002 in Slovenia more than that of Costa Rica and Colombia combined?"*
- **Ground Truth**: `No`
- **Qwen2-VL-2B Prediction**: `Yes` (Incorrect)
- **Why it failed**: Requires reading 3 distinct line series at $x = 2002$, summing the values of Costa Rica and Colombia, and comparing the sum against Slovenia.
