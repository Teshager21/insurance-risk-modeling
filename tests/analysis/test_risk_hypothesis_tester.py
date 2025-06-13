import pytest
import pandas as pd
from analysis.risk_hypthesis_tester import RiskHypothesisTester


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "TotalClaims": [100, 50, 0, 0, 200, 80, 0, 0],
            "TotalPremium": [1000, 700, 400, 300, 1200, 900, 500, 600],
            "Gender": [
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
            ],
            "Province": ["A", "A", "B", "B", "C", "C", "A", "B"],
            "PostalCode": [
                "1000",
                "1000",
                "2000",
                "2000",
                "3000",
                "3000",
                "1000",
                "2000",
            ],
        }
    )


def test_add_metrics(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()

    df = tester.df
    assert "ClaimFrequency" in df.columns
    assert "ClaimSeverity" in df.columns
    assert "Margin" in df.columns

    # Check known values
    assert df["ClaimFrequency"].sum() == 4  # 4 rows with TotalClaims > 0
    assert df["Margin"].iloc[0] == 900  # 1000 - 100
    assert pd.isna(df["ClaimSeverity"].iloc[2])  # Zero claims → NaN severity


def test_run_ttest(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    result = tester.run_ttest("Gender", "Margin")

    assert "p_value" in result
    assert "t_stat" in result
    assert isinstance(result["p_value"], float)


def test_run_chi2(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    result = tester.run_chi2("Gender", "ClaimFrequency")

    assert "chi2" in result
    assert "p_value" in result
    assert "group_details" in result
    assert isinstance(result["group_details"], dict)


def test_visualize_metric_runs(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    # We just check it doesn't crash
    tester.visualize_metric("Province", "Margin", kind="box")
    tester.visualize_metric("Province", "Margin", kind="bar")


def test_summarize_group_means(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    summary = tester.summarize_group_means("Province", "Margin")

    assert isinstance(summary, pd.DataFrame)
    assert "mean" in summary.columns
    assert "count" in summary.columns
    assert summary.shape[0] >= 1


def test_run_anova(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    result = tester.run_anova("Province", "Margin")

    assert "p_value" in result
    assert "f_stat" in result
    assert result["num_groups"] >= 2


def test_check_assumptions(sample_data):
    # Ensure enough group size by using min_group_size=2
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    result = tester.check_assumptions(
        group_col="Province", value_col="Margin", min_group_size=2
    )

    # Expect Shapiro results for each province
    assert isinstance(result, dict)
    assert "normality_p_values" in result
    assert "levene_p_value" in result
    # All three provinces should have a normality entry
    assert set(result["normality_p_values"].keys()) == {"A", "B", "C"}
    # Levene's p-value should be a float
    assert isinstance(result["levene_p_value"], float)


def test_kruskal_test(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    # Use min_group_size=2 to include all provinces
    result = tester.test_numeric_by_group_kruskal(
        "Province", "Margin", min_group_size=2
    )

    assert isinstance(result, dict)
    assert "p_value" in result and "statistic" in result
    # Should have compared 3 groups
    assert result["groups_used"] == 3
    assert isinstance(result["p_value"], float)


def test_two_group_difference(sample_data):
    tester = RiskHypothesisTester(sample_data)
    tester.add_metrics()
    result = tester.test_two_group_difference(group_col="Gender", numeric_col="Margin")

    # Validate output structure
    expected_keys = {
        "statistic",
        "p_value",
        "basic_interpretation",
        "detailed_interpretation",
        "group_1",
        "group_2",
        "mean_group_1",
        "mean_group_2",
        "diff_percent",
        "n1",
        "n2",
    }
    assert expected_keys.issubset(result.keys())
    # Numeric outputs
    assert isinstance(result["statistic"], float)
    assert isinstance(result["p_value"], float)
    # Counts should sum to total rows
    assert result["n1"] + result["n2"] == len(sample_data)
