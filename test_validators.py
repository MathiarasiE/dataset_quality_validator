"""
test_validators.py
------------------
Unit tests for the validation modules.
Run with: pytest test_validators.py -v
"""

import unittest
import pandas as pd
import numpy as np

import basic_validator
import ml_validator
import score as scorer
from backend.validator_pipeline import run_validation


class TestBasicValidator(unittest.TestCase):
    """Tests for basic_validator module."""

    def setUp(self):
        """Create sample datasets for testing."""
        # Clean dataset
        self.clean_df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "country": ["USA", "UK", "CA", "AU", "NZ"]
        })

        # Dataset with missing values
        self.missing_df = pd.DataFrame({
            "age": [25, 30, np.nan, 40, 45],
            "income": [50000, np.nan, 70000, 80000, 90000],
            "country": ["USA", "UK", "CA", "AU", "NZ"]
        })

        # Dataset with duplicates
        self.dup_df = pd.DataFrame({
            "age": [25, 30, 25, 40, 25],
            "income": [50000, 60000, 50000, 80000, 50000]
        })

    def test_check_missing_values_clean(self):
        """Missing values check should return 0 for clean dataset."""
        result = basic_validator.check_missing_values(self.clean_df)
        self.assertEqual(result["total_missing"], 0)
        self.assertEqual(result["missing_percentage"], 0.0)
        self.assertEqual(result["severity"], "none")

    def test_check_missing_values_dirty(self):
        """Missing values check should detect NaN values."""
        result = basic_validator.check_missing_values(self.missing_df)
        self.assertGreater(result["total_missing"], 0)
        self.assertGreater(result["missing_percentage"], 0)
        self.assertIn("age", [c["column"] for c in result["affected_columns"]])

    def test_check_duplicates_clean(self):
        """Duplicates check should return 0 for unique dataset."""
        result = basic_validator.check_duplicates(self.clean_df)
        self.assertEqual(result["duplicate_count"], 0)
        self.assertEqual(result["duplicate_percentage"], 0.0)

    def test_check_duplicates_dirty(self):
        """Duplicates check should detect duplicate rows."""
        result = basic_validator.check_duplicates(self.dup_df)
        self.assertGreater(result["duplicate_count"], 0)
        self.assertIn(result["severity"], ["low", "medium", "high"])  # Severity depends on % duplicates

    def test_check_data_types(self):
        """Data type check should handle mixed types."""
        mixed_df = pd.DataFrame({
            "numbers": [1, 2, "three", 4, 5],
            "text": ["a", "b", "c", "d", "e"]
        })
        result = basic_validator.check_data_types(mixed_df)
        self.assertGreater(result["mismatch_count"], 0)


class TestMLValidator(unittest.TestCase):
    """Tests for ml_validator module."""

    def setUp(self):
        """Create sample datasets."""
        np.random.seed(42)
        self.normal_df = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
            "label": np.random.choice([0, 1], 100)
        })

        # Dataset with outliers
        self.outlier_df = self.normal_df.copy()
        self.outlier_df.loc[0, "feature1"] = 100  # Add outlier

    def test_knn_outlier_detection(self):
        """KNN should detect outliers."""
        result = ml_validator.knn_outlier_detection(self.outlier_df)
        self.assertEqual(result["check"], "knn_outlier_detection")
        self.assertGreaterEqual(result["outlier_count"], 0)

    def test_kmeans_distribution_check(self):
        """K-Means should cluster data without errors."""
        result = ml_validator.kmeans_distribution_check(self.normal_df)
        self.assertEqual(result["check"], "kmeans_distribution")
        self.assertIsInstance(result["cluster_info"], list)

    def test_decision_tree_label_check(self):
        """Decision Tree should validate label consistency."""
        result = ml_validator.decision_tree_label_check(self.normal_df, label_col="label")
        self.assertEqual(result["check"], "decision_tree_label_consistency")
        self.assertIsNotNone(result["consistency_score"])

    def test_naive_bayes_distribution_check(self):
        """Naive Bayes should validate feature distributions."""
        result = ml_validator.naive_bayes_distribution_check(self.normal_df, label_col="label")
        self.assertEqual(result["check"], "naive_bayes_distribution")


class TestScorer(unittest.TestCase):
    """Tests for quality scoring system."""

    def setUp(self):
        """Create sample validation results."""
        self.results = [
            {"check": "missing_values", "severity": "low", "score_deduction": 2},
            {"check": "duplicates", "severity": "none", "score_deduction": 0},
            {"check": "outliers", "severity": "medium", "score_deduction": 5},
        ]

    def test_compute_quality_score(self):
        """Scoring should aggregate deductions correctly."""
        score_result = scorer.compute_quality_score(self.results)
        self.assertEqual(score_result["final_score"], 93.0)  # 100 - 2 - 5
        self.assertEqual(score_result["status"], "PASS")
        self.assertEqual(score_result["grade"], "A")

    def test_compute_quality_score_fail(self):
        """Scoring should fail when below threshold."""
        results = [
            {"check": "missing_values", "severity": "high", "score_deduction": 40},
        ]
        score_result = scorer.compute_quality_score(results)
        self.assertEqual(score_result["final_score"], 60.0)
        self.assertEqual(score_result["status"], "FAIL")


class TestValidationPipeline(unittest.TestCase):
    """Tests for the full validation pipeline."""

    def setUp(self):
        """Create test dataset."""
        self.test_df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "label": [0, 1, 0, 1, 0]
        })

    def test_run_validation_returns_report(self):
        """Pipeline should return a valid report."""
        report = run_validation(self.test_df, label_col="label", generate_report=False)
        self.assertIn("dataset_info", report)
        self.assertIn("validation_checks", report)
        self.assertIn("quality_score", report)
        self.assertIn("summary", report)

    def test_run_validation_dataset_info(self):
        """Dataset info should be correct."""
        report = run_validation(self.test_df, generate_report=False)
        self.assertEqual(report["dataset_info"]["rows"], 5)
        self.assertEqual(report["dataset_info"]["columns"], 3)

    def test_run_validation_has_quality_score(self):
        """Quality score should be between 0-100."""
        report = run_validation(self.test_df, generate_report=False)
        score = report["quality_score"]["final_score"]
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


if __name__ == "__main__":
    unittest.main()
