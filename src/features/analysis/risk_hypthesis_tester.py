import logging
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class RiskHypothesisTester:
    """
    A professional class for statistically validating insurance
    risk segmentation hypotheses.
    Supports metrics: Claim Frequency, Claim Severity, and Margin.
    """

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")

        self.df = df.copy()
        self._validate_required_columns()

    def _validate_required_columns(self):
        required = ["TotalClaims", "TotalPremium", "Sex", "Province", "zip"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def add_metrics(self):
        """Compute and add derived metrics to the dataset."""
        logger.info("Computing derived metrics...")
        self.df["ClaimFrequency"] = self.df["TotalClaims"].gt(0).astype(int)
        self.df["ClaimSeverity"] = self.df.apply(
            lambda x: (
                x["TotalClaims"] / x["ClaimFrequency"]
                if x["ClaimFrequency"]
                else np.nan
            ),
            axis=1,
        )
        self.df["Margin"] = self.df["TotalPremium"] - self.df["TotalClaims"]
        logger.info("Metrics added: ClaimFrequency, ClaimSeverity, Margin")

    def _check_groups(self, feature: str) -> Tuple[np.ndarray, np.ndarray]:
        unique_vals = self.df[feature].dropna().unique()
        if len(unique_vals) != 2:
            raise ValueError(
                f"Feature '{feature}' must have exactly 2 groups for A/B testing."
            )
        group_a, group_b = unique_vals
        return group_a, group_b

    def run_ttest(self, feature: str, metric: str) -> dict:
        """
        Run a t-test for a given feature and metric.
        Returns a dictionary with t-statistic and p-value.
        """
        logger.info(f"Running t-test for {metric} by {feature}...")

        group_a, group_b = self._check_groups(feature)
        a = self.df[self.df[feature] == group_a][metric].dropna()
        b = self.df[self.df[feature] == group_b][metric].dropna()

        stat, p_value = ttest_ind(a, b, equal_var=False)
        logger.info(
            f"T-test result (p={p_value:.5f}): "
            f"{'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} null"
        )

        return {
            "feature": feature,
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "t_stat": stat,
            "p_value": p_value,
        }

    def run_chi2(self, feature: str, binary_metric: str = "ClaimFrequency") -> dict:
        """
        Run chi-squared test for categorical feature vs
        binary metric (default: ClaimFrequency).
        Returns chi2, p-value, and degrees of freedom.
        """
        logger.info(f"Running chi-squared test for {feature} vs {binary_metric}...")
        contingency = pd.crosstab(self.df[feature], self.df[binary_metric])
        chi2, p, dof, _ = chi2_contingency(contingency)

        logger.info(
            f"Chi2 result (p={p:.5f}): "
            f"{'REJECT' if p < 0.05 else 'FAIL TO REJECT'} null"
        )
        return {
            "feature": feature,
            "metric": binary_metric,
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
        }

    def visualize_metric(self, feature: str, metric: str, kind: str = "box") -> None:
        """
        Visualize metric distribution across groups.
        :param feature: Feature to group by
        :param metric: Metric to visualize
        :param kind: 'box' or 'bar'
        """
        logger.info(f"Visualizing {metric} by {feature} with {kind} plot...")
        plt.figure(figsize=(8, 4))
        if kind == "box":
            sns.boxplot(data=self.df, x=feature, y=metric)
        elif kind == "bar":
            sns.barplot(data=self.df, x=feature, y=metric, estimator=np.mean, ci="sd")
        else:
            raise ValueError("Plot kind must be either 'box' or 'bar'.")

        plt.title(f"{metric} by {feature}")
        plt.tight_layout()
        plt.show()

    def summarize_group_means(self, feature: str, metric: str) -> pd.DataFrame:
        """
        Return a table of group means for a given metric.
        """
        logger.info(f"Summarizing group means of {metric} by {feature}...")
        return self.df.groupby(feature)[metric].agg(["mean", "std", "count"])
