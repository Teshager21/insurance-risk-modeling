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
from scipy.stats import f_oneway, shapiro, levene
import warnings

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
        """Plot box or bar chart for metric grouped by
        feature with wide layout for notebooks."""
        logger.info(f"Visualizing {metric} by {feature} ({kind} plot)...")

        # Set wide figure size
        plt.figure(figsize=(20, 6))  # 20 inches wide
        if kind == "box":
            sns.boxplot(data=self.df, x=feature, y=metric)
        elif kind == "bar":
            sns.barplot(data=self.df, x=feature, y=metric, estimator=np.mean, ci="sd")
        else:
            raise ValueError("Plot kind must be either 'box' or 'bar'.")

        plt.title(f"{metric} by {feature}", fontsize=16)
        plt.xticks(rotation=45, ha="right")  # Improved readability
        plt.grid(axis="y", linestyle="--", alpha=0.7)
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

    def export_markdown_report(
        self, results: dict, filename: str = "hypothesis_report.md"
    ):
        """
        Export test results to a markdown report.
        """
        from pathlib import Path

        p = results.get("p_value")
        stat = results.get("t_stat", results.get("chi2", None))
        result_str = results.get("result") or (
            "REJECT" if p is not None and p < 0.05 else "FAIL TO REJECT"
        )

        lines = [
            "# Hypothesis Test Report\n",
            "## Test Summary\n",
            f"- **Feature:** `{results.get('feature', 'N/A')}`",
            f"- **Metric:** `{results.get('metric', 'N/A')}`",
            f"- **Group A:** `{results.get('group_a', 'N/A')}`",
            f"- **Group B:** `{results.get('group_b', 'N/A')}`",
            (
                f"- **Statistic:** `{stat:.4f}`"
                if stat is not None
                else "- **Statistic:** `N/A`"
            ),
            f"- **p-value:** `{p:.4f}`" if p is not None else "- **p-value:** `N/A`",
            f"- **Result:** `{result_str}`\n",
            "## Interpretation\n",
            self.interpret_result(results),
        ]

        Path(filename).write_text("\n".join(lines))
        logger.info(f"Markdown report saved to {filename}")

    def interpret_result(self, result: dict) -> str:
        """
        Generate human-readable interpretation of hypothesis test.
        """
        p = result.get("p_value")
        feature = result.get("feature")
        metric = result.get("metric")
        interpretation = (
            f"We {'reject' if p is not None and p < 0.05 else 'fail to reject'}"
            f" the null hypothesis "
            f"that `{feature}` has no effect on `{metric}` (p = {p:.4f})."
        )

        # p = results.get("p_value")
        if p is not None and p > 0.05:

            interpretation += (
                f" This suggests that `{feature}` is a significant "
                f"factor affecting `{metric}`. "
                f"Consider using it in segmentation strategy or premium pricing."
            )
        else:
            interpretation += (
                f' This indicates no statistically "significant" impact of `{feature}`'
            )
            f"on `{metric}`."

        return interpretation

    def widget_ui(self):
        import ipywidgets as widgets
        from IPython.display import display, Markdown

        feature_dropdown = widgets.Dropdown(
            options=self.df.columns,
            description="Feature:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="50%"),
        )
        metric_dropdown = widgets.Dropdown(
            options=["ClaimFrequency", "ClaimSeverity", "Margin"],
            description="Metric:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="50%"),
        )

        output = widgets.Output()

        def run_test(_):
            with output:
                output.clear_output()
                try:
                    res = self.run_ttest(feature_dropdown.value, metric_dropdown.value)
                    display(Markdown(f"### Test Result\n- {res}"))
                    display(
                        Markdown(f"### Interpretation\n{self.interpret_result(res)}")
                    )
                    self.visualize_metric(feature_dropdown.value, metric_dropdown.value)
                except Exception as e:
                    print("Error:", e)

        button = widgets.Button(description="Run Test", button_style="success")
        button.on_click(run_test)

        display(widgets.VBox([feature_dropdown, metric_dropdown, button, output]))

    def run_anova(self, feature: str, metric: str) -> dict:
        """
        Perform one-way ANOVA for a numeric metric across multiple
        groups of a categorical feature.

        Args:
            feature (str): Categorical column to group by (e.g., Province).
            metric (str): Numeric metric column (e.g., Margin).

        Returns:
            dict: ANOVA result with F-statistic, p-value,
            feature, metric, and group info.
        """
        logger.info(f"Running ANOVA for {metric} by {feature}...")

        # Drop rows with missing values in relevant columns
        df_clean = self.df[[feature, metric]].dropna()
        groups = df_clean.groupby(feature)[metric].apply(list)

        if len(groups) < 2:
            logger.warning(f"Not enough groups in {feature} to perform ANOVA.")
            return {}

        try:
            f_stat, p_value = f_oneway(*groups)
            result = (
                "Reject Null Hypothesis"
                if p_value < 0.05
                else "Fail to Reject Null Hypothesis"
            )
            logger.info(f"ANOVA result (p={p_value:.5f}): {result}")

            return {
                "feature": feature,
                "metric": metric,
                "f_stat": f_stat,
                "p_value": p_value,
                "num_groups": len(groups),
                "result": result,
            }

        except Exception as e:
            logger.error(f"ANOVA failed for {feature} and {metric}: {e}")
            return {}

    def check_assumptions(
        self, group_col: str, value_col: str, min_group_size: int = 5
    ):
        """
        Checks the assumptions of ANOVA:
        1. Normality within groups (Shapiro-Wilk Test)
        2. Homogeneity of variances (Levene’s Test)

        Parameters:
        - group_col: Column to group by (e.g., 'Province', 'PostalCode', etc.)
        - value_col: Target numeric variable (e.g., 'ClaimFrequency', 'Margin')
        - min_group_size: Skip groups with fewer rows than this

        Returns:
        - Dict with normality and variance results
        """
        warnings.filterwarnings("ignore")

        print(f"\n📊 Checking Assumptions for '{value_col}' grouped by '{group_col}':")

        normality_results = {}
        groups = []

        print("\n🔍 Shapiro-Wilk Normality Test:")
        for name, group in self.df.groupby(group_col):
            if len(group) < min_group_size:
                print(f"  ⚠️ Skipping group '{name}' (n={len(group)}) < min_group_size")
                continue
            stat, p = shapiro(group[value_col])
            groups.append(group[value_col])
            normality_results[name] = p
            status = "✅ Normal" if p > 0.05 else "❌ Not Normal"
            print(f"  {name:<20}: p = {p:.4f} → {status}")

        print("\n🧪 Levene’s Test for Equal Variance:")
        if len(groups) >= 2:
            stat, p = levene(*groups)
            status = "✅ Equal Variance" if p > 0.05 else "❌ Variance Not Equal"
            print(f"  Levene’s p = {p:.4f} → {status}")
        else:
            print("  ⚠️ Not enough valid groups for Levene’s test")

        return {
            "normality_p_values": normality_results,
            "levene_p_value": p if len(groups) >= 2 else None,
        }
