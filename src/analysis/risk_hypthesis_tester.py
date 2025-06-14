import logging
from typing import Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from scipy.stats import f_oneway, shapiro, levene, kruskal, mannwhitneyu
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
        """
        Run Chi-squared test and display detailed group-level
        counts and proportions in a rich table.
        """
        logger.info(f"Running chi-squared test for '{feature}' vs '{binary_metric}'...")

        contingency = pd.crosstab(self.df[feature], self.df[binary_metric])
        chi2, p, dof, expected = chi2_contingency(contingency)
        conclusion = "REJECT" if p < 0.05 else "FAIL TO REJECT"

        # Print summary table
        self._print_result_table(
            title=f"Chi-Squared Test: {feature} vs {binary_metric}",
            rows=[
                ("Chi² Statistic", f"{chi2:.4f}"),
                ("Degrees of Freedom", f"{dof}"),
                ("P-Value", f"{p:.4f}"),
                ("Conclusion", f"{conclusion} null hypothesis"),
            ],
        )

        # Prepare detailed group table
        group_counts = contingency.sum(axis=1)
        group_proportions = contingency.div(group_counts, axis=0).round(4)

        table = Table(title=f"Group Details for '{feature}'")

        # Add columns: category + one column per binary_metric
        # category count + proportion + total
        table.add_column(feature, style="bold cyan")
        for col in contingency.columns:
            table.add_column(f"Count ({col})", justify="right")
            table.add_column(f"Prop ({col})", justify="right")
        table.add_column("Total", justify="right", style="bold")

        for category in contingency.index:
            counts = contingency.loc[category]
            proportions = group_proportions.loc[category]
            total = group_counts[category]
            row = [str(category)]
            for col in contingency.columns:
                row.append(str(counts[col]))
                row.append(f"{proportions[col]:.4f}")
            row.append(str(total))
            table.add_row(*row)

        console.print(table)

        return {
            "feature": feature,
            "metric": binary_metric,
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "conclusion": conclusion,
            "group_details": {
                category: {
                    "counts": contingency.loc[category].to_dict(),
                    "proportions": group_proportions.loc[category].to_dict(),
                    "total": group_counts[category],
                }
                for category in contingency.index
            },
            "contingency_table": contingency,
            "expected_freq": expected,
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

    def test_numeric_by_group_kruskal(
        self, group_col: str, numeric_col: str, min_group_size: int = 5
    ):
        """
        Performs Kruskal-Wallis H-test to compare
        distributions of a numeric variable across groups.

        Parameters:
        - group_col: Name of categorical grouping column.
        - numeric_col: Name of numeric column to compare.
        - min_group_size: Minimum samples in a group to include it.

        Returns:
        - dict with keys: p_value, statistic, groups_used
        """
        print(
            f"\n🔍 Kruskal-Wallis Test: '{numeric_col}' across '{group_col}' groups..."
        )

        # Group numeric values by group_col, filtering small groups
        grouped = [
            group[numeric_col]
            for _, group in self.df.groupby(group_col)
            if len(group) >= min_group_size
        ]
        included_groups = (
            self.df.groupby(group_col)
            .filter(lambda g: len(g) >= min_group_size)[group_col]
            .nunique()
        )

        if len(grouped) < 2:
            print("❌ Not enough valid groups to perform test.")
            return {"p_value": None, "statistic": None, "groups_used": 0}

        stat, p = kruskal(*grouped)
        print(f"✅ Kruskal-Wallis H-statistic = {stat:.4f}")
        print(
            f"📊 P-value = {p:.4f} → "
            + (
                "Reject H₀ (Differences exist)"
                if p < 0.05
                else "Fail to reject H₀ (No significant difference)"
            )
        )
        print(f"🧮 Groups compared: {included_groups}")

        return {"p_value": p, "statistic": stat, "groups_used": included_groups}

    def test_two_group_difference(
        self,
        group_col: str,
        numeric_col: str,
        group_values: Optional[Tuple[Any, Any]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Perform Mann–Whitney U test to compare distributions of
        a numeric variable between two groups,
        with a detailed business-friendly interpretation.

        Args:
            group_col (str): Column name for the grouping variable
            (must have exactly 2 unique values).
            numeric_col (str): Numeric column name to compare between groups.
            group_values (Optional[Tuple[Any, Any]]): Optional tuple
            specifying which two groups to compare.
                If None, will infer from unique values in group_col.
            alpha (float): Significance level for hypothesis testing (default 0.05).

        Returns:
            Dict[str, Any]: Dictionary containing test statistics, p-value,
            and detailed interpretation.
        """
        # Validate input DataFrame columns
        if group_col not in self.df.columns:
            raise ValueError(f"Grouping column '{group_col}' not found in DataFrame.")
        if numeric_col not in self.df.columns:
            raise ValueError(f"Numeric column '{numeric_col}' not found in DataFrame.")

        # Determine groups if not provided
        unique_vals = self.df[group_col].dropna().unique()
        if group_values is None:
            if len(unique_vals) != 2:
                raise ValueError(
                    f"Grouping column '{group_col}' "
                    f"must have exactly two unique non-null values "
                    f"for this test, found {len(unique_vals)}: {unique_vals}"
                )
            group_values = tuple(unique_vals)
        else:
            if len(group_values) != 2:
                raise ValueError(
                    "Parameter 'group_values' must be a tuple of exactly two values."
                )

        # Extract numeric data per group, drop missing values
        group1_data = self.df.loc[
            self.df[group_col] == group_values[0], numeric_col
        ].dropna()
        group2_data = self.df.loc[
            self.df[group_col] == group_values[1], numeric_col
        ].dropna()

        if group1_data.empty or group2_data.empty:
            return {
                "statistic": None,
                "p_value": None,
                "basic_interpretation": "Insufficient data in one or both groups.",
                "detailed_interpretation": None,
                "group_1": group_values[0],
                "group_2": group_values[1],
                "n1": len(group1_data),
                "n2": len(group2_data),
            }

        # Perform Mann-Whitney U test
        stat, p = mannwhitneyu(group1_data, group2_data, alternative="two-sided")

        # Basic hypothesis test interpretation
        basic_interp = (
            f"Reject H₀ (statistically significant difference, p={p:.4f})"
            if p < alpha
            else f"Fail to reject H₀ (no significant difference, p={p:.4f})"
        )

        # Calculate mean difference percentage relative to group 2 mean
        mean1 = group1_data.mean()
        mean2 = group2_data.mean()

        if np.isclose(mean2, 0):
            diff_pct = np.inf if not np.isclose(mean1, 0) else 0.0
        else:
            diff_pct = ((mean1 - mean2) / abs(mean2)) * 100

        # Detailed business interpretation
        if p < alpha:
            direction = "higher" if diff_pct > 0 else "lower"
            detailed_interp = (
                f"{group_values[0]} exhibits a "
                f"{abs(diff_pct):.1f}% {direction} average {numeric_col} "
                f"compared to {group_values[1]}, suggesting"
                f" a potential need for risk or pricing adjustment."
            )
        else:
            detailed_interp = (
                f"No statistically significant difference in"
                f" average {numeric_col} was detected between "
                f"{group_values[0]} and {group_values[1]} (p={p:.4f})."
            )

        return {
            "statistic": stat,
            "p_value": p,
            "basic_interpretation": basic_interp,
            "detailed_interpretation": detailed_interp,
            "group_1": group_values[0],
            "group_2": group_values[1],
            "mean_group_1": mean1,
            "mean_group_2": mean2,
            "diff_percent": diff_pct,
            "n1": len(group1_data),
            "n2": len(group2_data),
        }
