import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# Setup Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Rich Console
console = Console()


class RiskHypothesisTester:
    """
    A professional, modular, and visually-enhanced class for
    testing risk-based hypotheses in insurance datasets.
    Supports:
    - Claim Frequency
    - Claim Severity
    - Margin
    """

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")
        self.df = df.copy()
        self.results: Dict = {}
        self._validate_required_columns()

    def _validate_required_columns(self):
        required = ["TotalClaims", "TotalPremium", "Gender", "Province", "PostalCode"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def add_metrics(self):
        """Add ClaimFrequency, ClaimSeverity, and Margin to the DataFrame."""
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
        logger.info("Added metrics: ClaimFrequency, ClaimSeverity, Margin")

    def _check_binary_group(self, feature: str) -> Tuple[str, str]:
        groups = self.df[feature].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"{feature} must have exactly two groups for A/B testing.")
        return groups[0], groups[1]

    def run_ttest(self, feature: str, metric: str) -> Dict:
        """Run Welch's t-test for a metric grouped by a binary feature."""
        logger.info(f"Running t-test for {metric} by {feature}...")
        group_a, group_b = self._check_binary_group(feature)

        a = self.df[self.df[feature] == group_a][metric].dropna()
        b = self.df[self.df[feature] == group_b][metric].dropna()

        stat, p_value = ttest_ind(a, b, equal_var=False)
        result = "REJECT" if p_value < 0.05 else "FAIL TO REJECT"

        self._print_result_table(
            title=f"T-Test Result: {feature} vs {metric}",
            rows=[
                ("Group A", str(group_a)),
                ("Group B", str(group_b)),
                ("T-Statistic", f"{stat:.4f}"),
                ("P-Value", f"{p_value:.4f}"),
                ("Conclusion", result + " null hypothesis"),
            ],
        )

        return {
            "feature": feature,
            "metric": metric,
            "group_a": group_a,
            "group_b": group_b,
            "t_stat": stat,
            "p_value": p_value,
            "conclusion": result,
        }

    def run_chi2(self, feature: str, binary_metric: str = "ClaimFrequency") -> Dict:
        """Run Chi-squared test for categorical feature vs binary metric."""
        logger.info(f"Running chi-squared test for {feature} vs {binary_metric}...")
        contingency = pd.crosstab(self.df[feature], self.df[binary_metric])
        chi2, p, dof, _ = chi2_contingency(contingency)

        result = "REJECT" if p < 0.05 else "FAIL TO REJECT"

        self._print_result_table(
            title=f"Chi-Squared Test: {feature} vs {binary_metric}",
            rows=[
                ("Chi² Statistic", f"{chi2:.4f}"),
                ("Degrees of Freedom", f"{dof}"),
                ("P-Value", f"{p:.4f}"),
                ("Conclusion", result + " null hypothesis"),
            ],
        )

        return {
            "feature": feature,
            "metric": binary_metric,
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "conclusion": result,
        }

    def visualize_metric(self, feature: str, metric: str, kind: str = "box") -> None:
        """Plot box or bar chart for metric grouped by feature."""
        logger.info(f"Visualizing {metric} by {feature} ({kind} plot)...")
        plt.figure(figsize=(8, 5))
        if kind == "box":
            sns.boxplot(data=self.df, x=feature, y=metric)
        elif kind == "bar":
            sns.barplot(data=self.df, x=feature, y=metric, estimator=np.mean, ci="sd")
        else:
            raise ValueError("Plot kind must be either 'box' or 'bar'.")
        plt.title(f"{metric} by {feature}", fontsize=14)
        plt.xticks(rotation=15)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()

    def summarize_group_means(self, feature: str, metric: str) -> pd.DataFrame:
        """Return a group summary table."""
        summary = self.df.groupby(feature)[metric].agg(["mean", "std", "count"])
        console.print(Panel(f"[bold cyan]Summary of {metric} by {feature}[/bold cyan]"))
        console.print(summary)
        return summary

    def _print_result_table(self, title: str, rows: list):
        """Helper to print a colorful table using Rich."""
        table = Table(title=title, title_style="bold magenta")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="green")
        for name, value in rows:
            table.add_row(str(name), str(value))
        console.print(table)
