from typing import Optional, List, Dict, Union
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataQualityUtils:
    """A utility class for performing data quality checks and
    cleaning operations on a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Raises:
            TypeError: If input is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df.copy()

    def clean_column_names(self) -> pd.DataFrame:
        """
        Standardize column names to lowercase and replace spaces with underscores.

        Returns:
            pd.DataFrame: DataFrame with cleaned column names.
        """
        self.df.columns = (
            self.df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        return self.df

    def drop_redundant_columns(self) -> pd.DataFrame:
        """
        Drop commonly redundant columns such as 'unnamed: 0' and exact duplicates.

        Returns:
            pd.DataFrame: DataFrame with redundant columns removed.
        """
        if "unnamed:_0" in self.df.columns:
            self.df.drop(columns=["unnamed:_0"], inplace=True)
            logger.info("Dropped 'unnamed:_0' column")

        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        return self.df

    def clean_dataframe(self) -> pd.DataFrame:
        """
        Apply full cleaning pipeline.

        Returns:
            pd.DataFrame: Fully cleaned DataFrame.
        """
        self.clean_column_names()
        self.drop_redundant_columns()
        self.drop_constant_columns(verbose=False)
        return self.df

    def columns_with_significant_missing_values(
        self, threshold: float = 5.0
    ) -> pd.DataFrame:
        """
        Identify columns with missing values above a given threshold.

        Args:
            threshold (float): Minimum percentage of missing values to flag a column.

        Returns:
            pd.DataFrame: DataFrame of columns exceeding the missing value threshold.
        """
        missing_counts = self.df.isna().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        significant = missing_percent[missing_percent > threshold]

        return pd.DataFrame(
            {
                "#missing_values": missing_counts[significant.index],
                "percentage": significant.map(lambda x: f"{x:.2f}%"),
            }
        ).sort_values(by="#missing_values", ascending=False)

    def summary(self) -> pd.DataFrame:
        """
        Provide summary of missing values in the DataFrame.

        Returns:
            pd.DataFrame: Summary with count and percentage of missing values.
        """
        missing_counts = self.df.isna().sum()
        missing_percent = (missing_counts / len(self.df)) * 100

        return pd.DataFrame(
            {
                "#missing_values": missing_counts,
                "percentage": missing_percent.map(lambda x: f"{x:.2f}%"),
            }
        ).sort_values(by="#missing_values", ascending=False)

    def check_duplicates(self) -> int:
        """
        Count duplicate rows.

        Returns:
            int: Number of duplicate rows.
        """
        return self.df.duplicated().sum()

    def count_duplicates(self) -> int:
        """
        Alias for check_duplicates.

        Returns:
            int: Number of duplicate rows.
        """
        return self.check_duplicates()

    def find_invalid_values(
        self, additional_invalids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[int, pd.Series]]]:
        """
        Find invalid values in object columns (e.g., empty strings, 'NA', 'null').

        Args:
            additional_invalids (List[str], optional):
            List of additional invalid strings.

        Returns:
            dict: Summary of invalid values for each affected column.
        """
        if additional_invalids is None:
            additional_invalids = ["NA", "null", "NULL", "-", "N/A"]

        invalid_summary = {}
        for col in self.df.select_dtypes(include="object").columns:
            mask = self.df[col].astype(str).str.strip().isin(["", *additional_invalids])
            count = mask.sum()
            if count > 0:
                invalid_summary[col] = {
                    "count": count,
                    "examples": self.df.loc[mask, col].head(5),
                }
        return invalid_summary

    def convert_columns_to_datetime(
        self, columns: Optional[List[str]] = None, errors: str = "coerce"
    ) -> pd.DataFrame:
        """
        Convert specified or inferred date/time columns to datetime format.

        Args:
            columns (List[str], optional): Columns to convert. If None, infer columns.
            errors (str): Error handling strategy ('coerce', 'raise', 'ignore').

        Returns:
            pd.DataFrame: DataFrame with converted datetime columns.
        """
        if columns is None:
            columns = [
                col
                for col in self.df.columns
                if "date" in col.lower() or "time" in col.lower()
            ]

        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found.")
                continue

            original_non_null = self.df[col].notna().sum()

            self.df[col] = self.df[col].astype(str).str.strip()
            self.df[col] = self.df[col].replace(
                ["", "nan", "null", "None", "NaT", "N/A"], pd.NA
            )
            self.df[col] = pd.to_datetime(self.df[col], errors=errors, utc=True)

            converted = self.df[col].notna().sum()
            logger.info(
                f"[{col}] Converted: {converted}/{original_non_null} "
                f"({original_non_null - converted} became NaT)"
            )

        return self.df

    def drop_constant_columns(self, verbose: bool = True) -> pd.DataFrame:
        """
        Drops columns with only one unique value in the DataFrame.

        Parameters:
            verbose (bool): Whether to log dropped column names.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            if not isinstance(verbose, bool):
                logger.warning(
                    f"[drop_constant_columns] Expected boolean for 'verbose', "
                    f"got {type(verbose)}. Defaulting to True."
                )
                verbose = True  # or False, depending on your preference

            constant_cols = [
                col
                for col in self.df.columns
                if self.df[col].nunique(dropna=False) == 1
            ]

            if constant_cols:
                self.df.drop(columns=constant_cols, inplace=True)
                if verbose:
                    logger.info(f"Dropped constant columns: {constant_cols}")
            else:
                if verbose:
                    logger.info("No constant columns found.")

            return self.df

        except Exception:
            logger.exception("Error while dropping constant columns.")
            raise

    def drop_columns(
        self, df: pd.DataFrame, columns_to_drop: list, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame from which to drop columns.
        columns_to_drop : list
            A list of column names to be dropped.
        inplace : bool, optional (default=False)
            If True, modifies the original DataFrame. If False, returns a modified copy.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with the specified columns removed (if inplace=False).

        Raises:
        -------
        ValueError:
            If none of the specified columns are found in the DataFrame.
        """
        if not isinstance(columns_to_drop, list):
            raise TypeError("columns_to_drop must be a list of column names.")

        existing_cols = df.columns.tolist()
        missing_cols = [col for col in columns_to_drop if col not in existing_cols]

        if len(missing_cols) == len(columns_to_drop):
            raise ValueError(
                "None of the specified columns were found in the DataFrame."
            )

        valid_cols_to_drop = [col for col in columns_to_drop if col in existing_cols]

        if inplace:
            df.drop(columns=valid_cols_to_drop, inplace=True)
            return None
        else:
            return df.drop(columns=valid_cols_to_drop)

    def impute_value_in_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        target_value,
        impute_value,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Checks if a target value exists in a specified column and
        imputes it with a given value.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to process.
        column_name : str
            The name of the column to check and impute.
        target_value :
            The value to search for in the column.
        impute_value :
            The value to replace the target value with.
        inplace : bool, optional (default=False)
            If True, modifies the original DataFrame. If False, returns a modified copy.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with values imputed (if inplace=False), else None.

        Raises:
        -------
        ValueError:
            If the specified column is not in the DataFrame.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        if df[column_name].isin([target_value]).any():
            if inplace:
                df[column_name] = df[column_name].replace(target_value, impute_value)
                return None
            else:
                df_copy = df.copy()
                df_copy[column_name] = df_copy[column_name].replace(
                    target_value, impute_value
                )
                return df_copy
        else:
            if not inplace:
                return df.copy()
            return None  # no change made

    def impute_missing_values(
        self, df: pd.DataFrame, columns: list, impute_value, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Imputes missing (NaN) values in the specified columns with a given value.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to process.
        columns : list
            List of column names to impute.
        impute_value :
            The value to use for imputing missing entries.
        inplace : bool, optional (default=False)
            If True, modifies the original DataFrame. If False, returns a modified copy.

        Returns:
        --------
        pd.DataFrame or None
            Modified DataFrame if inplace is False; otherwise, None.

        Raises:
        -------
        ValueError:
            If any specified column is not found in the DataFrame.
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following columns are not in the DataFrame: {missing_cols}"
            )

        if inplace:
            df[columns] = df[columns].fillna(impute_value)
            return None
        else:
            df_copy = df.copy()
            df_copy[columns] = df_copy[columns].fillna(impute_value)
            return df_copy

    def drop_rows_with_missing_values(
        self, df: pd.DataFrame, columns: list, inplace: bool = False
    ) -> pd.DataFrame:
        """
        Drops rows where any of the specified columns contain NaN (missing) values.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to process.
        columns : list
            List of column names to check for missing values.
        inplace : bool, optional (default=False)
            If True, modifies the original DataFrame. If False, returns a modified copy.

        Returns:
        --------
        pd.DataFrame or None
            Modified DataFrame if inplace is False; otherwise, None.

        Raises:
        -------
        ValueError:
            If any of the specified columns are not in the DataFrame.
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following columns are not in the DataFrame: {missing_cols}"
            )

        if inplace:
            df.dropna(subset=columns, inplace=True)
            return None
        else:
            return df.dropna(subset=columns)

    @staticmethod
    def detect_outliers_iqr(
        df: pd.DataFrame, columns: List[str], iqr_multiplier: float = 1.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in specified numeric columns of a DataFrame
        using the Interquartile Range (IQR) method.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame.
        columns : List[str]
            List of column names to check for outliers.
        iqr_multiplier : float, optional
            The multiplier for the IQR to determine the bounds (default is 1.5).

        Returns:
        -------
        Dict[str, pd.DataFrame]
            A dictionary where keys are column names and values
            are DataFrames containing outlier rows for that column.
        """
        outliers = {}

        for col in columns:
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR

            outlier_rows = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            if not outlier_rows.empty:
                outliers[col] = outlier_rows

        return outliers

    @staticmethod
    def detect_outliers_iqr_with_boxplot(
        df: pd.DataFrame,
        columns: List[str],
        iqr_multiplier: float = 1.5,
        show_plots: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers using the IQR method and optionally show boxplots.

        Parameters:
        - df: DataFrame to check.
        - columns: List of column names to inspect.
        - iqr_multiplier: Sensitivity of IQR method.
        - show_plots: If True, display plots.
        - save_dir: Directory path to save figures. If None, plots are not saved.

        Returns:
        - Dict of column names to outlier DataFrames.
        """
        outliers = {}

        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            col_series = df[col].dropna()
            if col_series.nunique() < 2:
                continue  # Not enough data to compute a boxplot

            Q1 = col_series.quantile(0.25)
            Q3 = col_series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR

            mask = (df[col] < lower) | (df[col] > upper)
            if mask.any():
                outliers[col] = df[mask]

            # Only plot if data is valid
            if (show_plots or save_dir) and col_series.notna().sum() > 0:
                plt.figure(figsize=(6, 1.5))
                sns.boxplot(x=col_series, color="skyblue", fliersize=4)
                plt.title(f"Boxplot for {col}")
                plt.tight_layout()

                if save_dir:
                    plt.savefig(f"{save_dir}/{col}_boxplot.png")
                if show_plots:
                    plt.show()
                plt.close()

        return outliers
