import pytest
import pandas as pd
from preprocessing.eda import EDAUtils, MultivariateAnalysis

# tests for EDAUtils and MultivariateAnalysis classes


@pytest.fixture
def sample_mixed_data(tmp_path):
    # Create a sample DataFrame with numeric, categorical, and datetime-like columns
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [5, 4, 3, 2, 1],
            "cat": ["a", "b", "a", "b", "c"],
            "date_str": ["2020-01-01", "2020-02-01", "invalid", None, "2020-05-01"],
        }
    )
    return df


def test_univariate_analysis_inline(sample_mixed_data, capsys):
    eda = EDAUtils(sample_mixed_data)
    # Should not raise and show plots inline
    eda.univariate_analysis(save_dir=None, bins=3, top_n=2)
    # No exception means pass


def test_univariate_analysis_save(tmp_path, sample_mixed_data):
    eda = EDAUtils(sample_mixed_data)
    save_dir = tmp_path / "plots"
    eda.univariate_analysis(save_dir=str(save_dir), bins=2, top_n=2)
    # Check that files for each column are saved
    for col in sample_mixed_data.columns:
        img_file = save_dir / f"{col}_univariate.png"
        assert img_file.exists(), f"Expected plot file not found: {img_file}"


def test_pairplot_and_heatmap_and_pca(tmp_path, sample_mixed_data):
    # MultivariateAnalysis smoke tests
    mva = MultivariateAnalysis(sample_mixed_data)
    # Test pairplot inline and save
    save_pp = tmp_path / "pairplot.png"
    mva.pairplot(features=["num1", "num2"], hue="cat", save_path=str(save_pp))
    assert save_pp.exists()

    # Test correlation heatmap inline and save
    save_hm = tmp_path / "heatmap.png"
    mva.correlation_heatmap(
        features=["num1", "num2"], annot=False, save_path=str(save_hm)
    )
    assert save_hm.exists()

    # Test PCA analysis inline and save
    save_pca = tmp_path / "pca.png"
    mva.pca_analysis(
        features=["num1", "num2"],
        hue="cat",
        n_components=2,
        scale=True,
        save_path=str(save_pca),
    )
    assert save_pca.exists()


def test_init_validation():
    # Passing non-DataFrame to __init__ should raise
    with pytest.raises(ValueError):
        EDAUtils(123)
    with pytest.raises(ValueError):
        MultivariateAnalysis("not a df")


def test_univariate_skip_unsupported(tmp_path):
    # Create DataFrame with unsupported dtype (e.g., list)
    df = pd.DataFrame({"col": [[1, 2], [3, 4], [5, 6]]})
    eda = EDAUtils(df)
    # Should not crash
    eda.univariate_analysis(save_dir=None)
