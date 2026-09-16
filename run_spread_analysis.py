#!/usr/bin/env python3
"""
Spread & Variance Analysis Script (Per Assignment Specification):
Measures performance across meaning-preserving prompt rewordings and runs
to report the confidence spread [min_acc, max_acc] and format the Page 3 defense claim.
"""

import os
import json
import glob
import argparse


def analyze_spread(results_dir: str):
    summary_files = glob.glob(os.path.join(results_dir, "summary_*.json"))
    if not summary_files:
        print(f"No summary files found in {results_dir}")
        return

    print("=" * 75)
    print("CHARTQA BENCHMARK RUNS & SPREAD ANALYSIS")
    print("=" * 75)

    runs = []
    for fpath in sorted(summary_files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            runs.append(data)

    print(f"{'Run / Template':<22} | {'Seed':<5} | {'Strict Acc (%)':<14} | {'Harness Acc (%)':<16} | {'Human Acc':<10} | {'Aug Acc':<10}")
    print("-" * 88)

    strict_scores = []
    harness_scores = []

    for r in runs:
        m = r["metrics"]
        t = r.get("template", "unknown")
        s = r.get("seed", 42)
        strict_acc = m["overall_relaxed_accuracy_strict"]
        harness_acc = m["overall_relaxed_accuracy_harness"]
        h_acc = m["chartqa_human_strict"]
        a_acc = m["chartqa_augmented_strict"]

        strict_scores.append(strict_acc)
        harness_scores.append(harness_acc)

        print(f"{t:<22} | {s:<5} | {strict_acc:<14.2f} | {harness_acc:<16.2f} | {h_acc:<10.2f} | {a_acc:<10.2f}")

    print("-" * 88)

    min_strict = min(strict_scores)
    max_strict = max(strict_scores)
    avg_strict = sum(strict_scores) / len(strict_scores)
    spread_strict = max_strict - min_strict

    min_harness = min(harness_scores)
    max_harness = max(harness_scores)
    avg_harness = sum(harness_scores) / len(harness_scores)
    spread_harness = max_harness - min_harness

    print(f"\nStrict Scorer Spread:   Mean = {avg_strict:.2f}% | Range = [{min_strict:.2f}%, {max_strict:.2f}%] | Spread = {spread_strict:.2f}%")
    print(f"Harness Scorer Spread:  Mean = {avg_harness:.2f}% | Range = [{min_harness:.2f}%, {max_harness:.2f}%] | Spread = {spread_harness:.2f}%")
    print(f"Harness Delta:          +{avg_harness - avg_strict:.2f}% absolute improvement from regex post-parsing")
    print("=" * 75)

    print("\nDEFENSE CLAIM (PAGE 3 ASSIGNMENT REQUIREMENT):")
    rep = runs[0]
    claim = (
        f"\"Model {rep['model_id']} at revision 'main', fp16 on GPU 1 (RTX 3090 Ti / V100 profile), "
        f"greedy, template '{rep.get('template', 'direct')}', scored {rep['metrics']['overall_relaxed_accuracy_strict']:.2f}% "
        f"relaxed accuracy (range [{min_strict:.2f}%, {max_strict:.2f}%] over {len(runs)} meaning-preserving rewordings); "
        f"the literature reports 78.5% for GPT-4V and 90.8% for Claude-3.5-Sonnet. "
        f"The difference is confounded with model scale (2B vs >1T), visual encoder resolution tokens, "
        f"pretraining data mixture, and chain-of-thought instruction tuning.\""
    )
    print(claim)
    print("=" * 75 + "\n")

    # Save aggregated spread report
    spread_summary = {
        "num_runs": len(runs),
        "strict_mean": round(avg_strict, 2),
        "strict_range": [round(min_strict, 2), round(max_strict, 2)],
        "strict_spread": round(spread_strict, 2),
        "harness_mean": round(avg_harness, 2),
        "harness_range": [round(min_harness, 2), round(max_harness, 2)],
        "harness_spread": round(spread_harness, 2),
        "defense_claim": claim,
        "runs": runs,
    }
    with open(os.path.join(results_dir, "spread_summary.json"), "w", encoding="utf-8") as f:
        json.dump(spread_summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    analyze_spread(args.results_dir)
