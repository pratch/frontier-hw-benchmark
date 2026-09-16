#!/usr/bin/env python3
"""
Unit tests for the ChartQA Relaxed Accuracy scorer and harness parser.
"""

import unittest
from evaluate_chartqa import relaxed_correctness, harness_enhanced_parse, to_float


class TestScorer(unittest.TestCase):
    def test_numeric_within_5_percent(self):
        # 100 vs 97 (3% relative difference -> True)
        self.assertTrue(relaxed_correctness("97", "100"))
        # 100 vs 104.5 (4.5% relative difference -> True)
        self.assertTrue(relaxed_correctness("104.5", "100"))
        # With percentage signs
        self.assertTrue(relaxed_correctness("96%", "100%"))
        # With dollar signs and commas
        self.assertTrue(relaxed_correctness("$1,000", "$980"))

    def test_numeric_outside_5_percent(self):
        # 100 vs 90 (10% difference -> False)
        self.assertFalse(relaxed_correctness("90", "100"))
        # 50 vs 55 (10% difference -> False)
        self.assertFalse(relaxed_correctness("55", "50"))

    def test_zero_handling(self):
        # 0 vs 0 -> True
        self.assertTrue(relaxed_correctness("0", "0"))
        # 0 vs 0.04 -> within 0.05 absolute tolerance -> True
        self.assertTrue(relaxed_correctness("0.04", "0"))
        # 0 vs 1.0 -> False
        self.assertFalse(relaxed_correctness("1.0", "0"))

    def test_text_exact_match(self):
        self.assertTrue(relaxed_correctness("Bar chart.", "bar chart"))
        self.assertTrue(relaxed_correctness("Yes", "yes."))
        self.assertTrue(relaxed_correctness("United States", "united states"))
        self.assertFalse(relaxed_correctness("Line chart", "Bar chart"))

    def test_harness_cleaning(self):
        # Cleans introductory conversational wrappers
        self.assertEqual(harness_enhanced_parse("The answer is 42.5%."), "42.5%")
        self.assertEqual(harness_enhanced_parse("Based on the chart, it is 120"), "120")
        self.assertEqual(harness_enhanced_parse("Approximately 35"), "35")
        self.assertEqual(harness_enhanced_parse("42"), "42")


if __name__ == "__main__":
    unittest.main()
