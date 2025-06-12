import pandas as pd
import logging

# from typing import #Optional
from IPython.display import display, HTML

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Optional: StreamHandler with formatter
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataInspectionUtils:
    """
    A utility class for inspecting pandas DataFrames.
    Provides methods to display unique values and metadata for all columns.
    """

    @staticmethod
    def display_column_uniques(df: pd.DataFrame, max_items: int = 10) -> None:
        """
        Display a summary of unique values for each column in a DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to inspect.

        max_items : int, optional, default=10
            Maximum number of unique values to display per column.

        Returns:
        -------
        pd.DataFrame
            A summary DataFrame containing:
            - Column name
            - Data type
            - Unique count
            - Sample of unique values (up to max_items)
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")

            if df.empty:
                logger.warning("The DataFrame is empty. No unique values to display.")
                return None

            logger.info("Computing unique values for each column...")
            summary = []

            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                total_uniques = len(unique_vals)

                if total_uniques > max_items:
                    sample_vals = list(unique_vals[:max_items]) + ["..."]
                else:
                    sample_vals = list(unique_vals)

                summary.append(
                    {
                        "Column": col,
                        "Data Type": str(df[col].dtype),
                        "Unique Count": total_uniques,
                        f"Unique Sample (max {max_items})": sample_vals,
                    }
                )

            summary_df = pd.DataFrame(summary)

            try:
                display(HTML(summary_df.to_html(index=False)))
            except Exception as e:
                logger.warning(f"Could not render HTML table in notebook: {e}")
                print(summary_df)

            logger.info("Unique values displayed successfully.")
            return None

        except Exception as e:
            logger.error(f"Failed to display column uniques: " f"{e}", exc_info=True)
            return None

    @staticmethod
    def get_constant_columns(df: pd.DataFrame, return_df: bool = False) -> None:
        """
        Identify columns in the DataFrame that contain only a single unique value.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to analyze.
        return_df : bool, optional
            If True, returns the results as a DataFrame; else, prints them.

        Returns
        -------
        Optional[pd.DataFrame]
            A DataFrame containing columns with a single unique value
            and the constant value itself,or None if `return_df` is False.
        """
        try:
            logger.info("Checking for columns with only one unique value...")

            constant_cols = []
            constant_vals = []

            for col in df.columns:
                uniques = df[col].nunique(dropna=False)
                if uniques == 1:
                    constant_cols.append(col)
                    constant_vals.append(df[col].unique()[0])

            if not constant_cols:
                logger.info("No constant columns found.")
                print("✅ No columns with a single unique value found.")
                return None

            result = pd.DataFrame(
                {"column": constant_cols, "constant_value": constant_vals}
            )

            logger.info(f"Found {len(constant_cols)} constant columns.")

            if return_df:
                return result
            else:
                from IPython.display import display, HTML

                display(HTML(result.to_html(index=False)))

        except Exception as e:
            logger.error(f"Error identifying constant columns: {e}")
            raise
