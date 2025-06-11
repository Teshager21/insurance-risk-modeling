from typing import Optional, List, Dict, Union
import pandas as pd
import logging

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

    # def drop_constant_columns(self, verbose: bool = True) -> pd.DataFrame:
    #     """
    #     Drop columns with only one unique value.

    #     Args:
    #         verbose (bool): Whether to log the names of
    #         dropped columns. Default is True.

    #     Returns:
    #         pd.DataFrame: DataFrame with constant columns removed.
    #     """
    #     try:
    #         constant_cols = [
    #             col
    #             for col in self.df.columns
    #             if self.df[col].nunique(dropna=False) == 1
    #         ]

    #         if constant_cols:
    #             self.df.drop(columns=constant_cols, inplace=True)
    #             if verbose:
    #                 msg = f"Dropped {len(constant_cols)} constant column(s): "
    #                 f"{', '.join(constant_cols)}"
    #                 logger.info(msg)
    #         elif verbose:
    #             logger.info("✅ No constant columns to drop.")

    #     except Exception as e:
    #         logger.error(f"Error dropping constant columns: {e}")
    #         raise

    #     return self.df

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
