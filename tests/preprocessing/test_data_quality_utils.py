import pytest
import pandas as pd
from preprocessing.data_quality_utils import DataQualityUtils


@pytest.fixture
def quality_data():
    df = pd.DataFrame(
        {
            "A B": [1, 2, 3],
            "unnamed:_0": [0, 1, 2],
            "dup": [1, 1, 1],
            "val": [None, "NA", "x"],
            "date_col": ["2020-01-01", "invalid", "2020-03-01"],
            "num": [1, 100, 1000],
        }
    )
    return df


def test_clean_and_drop(quality_data):
    dq = DataQualityUtils(quality_data)
    df1 = dq.clean_column_names()
    assert "a_b" in df1.columns and "unnamed:_0" in df1.columns

    df2 = dq.drop_redundant_columns()
    assert "unnamed:_0" not in df2.columns
    assert "dup" in df2.columns


def test_missing_summary(quality_data):
    dq = DataQualityUtils(quality_data)
    # sig = dq.columns_with_significant_missing_values(threshold=10)
    full = dq.summary()
    assert "val" in full.index
    assert "#missing_values" in full.columns


def test_duplicates(quality_data):
    dq = DataQualityUtils(quality_data)
    # no duplicate rows
    assert dq.check_duplicates() == 0
    assert dq.count_duplicates() == 0


def test_find_invalid_values(quality_data):
    dq = DataQualityUtils(quality_data)
    invalid = dq.find_invalid_values()
    assert "val" in invalid
    assert invalid["val"]["count"] >= 1


def test_convert_datetime(quality_data):
    dq = DataQualityUtils(quality_data)
    df = dq.convert_columns_to_datetime(columns=["date_col"], errors="coerce")
    assert pd.api.types.is_datetime64_any_dtype(df["date_col"])


def test_drop_constant(quality_data):
    dq = DataQualityUtils(quality_data)
    df = dq.drop_constant_columns()
    assert "dup" not in df.columns


def test_drop_columns_and_impute(quality_data):
    dq = DataQualityUtils(quality_data)
    # drop existing
    df_new = dq.drop_columns(quality_data, ["num"], inplace=False)
    assert "num" not in df_new.columns

    # impute value
    df_imp = dq.impute_value_in_column(quality_data, "val", "NA", "IM")
    assert "IM" in df_imp["val"].values


def test_impute_missing_and_drop_rows(quality_data):
    dq = DataQualityUtils(quality_data)
    df_imp = dq.impute_missing_values(quality_data, ["val"], impute_value="X")
    assert df_imp["val"].isna().sum() == 0
    df_drop = dq.drop_rows_with_missing_values(quality_data, ["val"])
    assert df_drop["val"].isna().sum() == 0


def test_outliers_iqr(quality_data):
    dq = DataQualityUtils(quality_data)
    out = dq.detect_outliers_iqr(quality_data, ["num"])
    assert isinstance(out, dict)


def test_outliers_boxplot(quality_data):
    dq = DataQualityUtils(quality_data)
    # smoke test, no errors
    out2 = dq.detect_outliers_iqr_with_boxplot(quality_data, ["num"], show_plots=False)
    assert isinstance(out2, dict)


def test_infer_gender(quality_data):
    df = quality_data.copy()
    df["Gender"] = ["not specified", "Female", "not specified"]
    df["Title"] = ["Mr", "Ms", "Dr"]
    dq = DataQualityUtils(df)
    df2 = dq.infer_gender_from_title()
    assert "Gender_Inferred" in df2.columns


def test_infer_dr_gender(quality_data):
    df = quality_data.copy()
    df["Gender"] = ["not specified", "Male", "Female"]
    df["Title"] = ["Dr", "Dr", "Other"]
    dq = DataQualityUtils(df)
    df3 = dq.infer_dr_gender_by_majority()
    assert df3["Gender"].isin(["Male", "Female"]).all()
