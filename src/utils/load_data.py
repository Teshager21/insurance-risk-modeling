import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataLoader:
    """
    A flexible data loader for CSV, Excel, and Parquet files.
    Supports local paths and URLs (including S3 URLs if boto3 is configured).
    """

    def __init__(self, source: str):
        self.source = source

    def load(self, file_type: str = "csv", **kwargs) -> pd.DataFrame:
        """
        Load data from the specified source.

        Parameters:
        - file_type: str - Type of the file ('csv', 'excel', 'parquet').
        - kwargs: Additional keyword arguments passed to the Pandas
                  reader function.

        Returns:
        - pd.DataFrame containing the loaded data.

        Raises:
        - ValueError for unsupported file types or failed reads.
        """
        try:
            logger.info(f"Loading data from: {self.source} (type={file_type})")

            if file_type == "csv":
                df = pd.read_csv(self.source, **kwargs)
            elif file_type == "excel":
                df = pd.read_excel(self.source, **kwargs)
            elif file_type == "parquet":
                df = pd.read_parquet(self.source, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data from {self.source}: {e}")
            raise


def load_local_data(
    relative_path: str, file_type: str = "csv", **kwargs
) -> pd.DataFrame:
    """
    Load data from a file in the local 'data/raw' directory.

    Example usage:
    >>> df = load_local_data("stock_data.csv", file_type="csv")

    Parameters:
    - relative_path: Path relative to the 'data/raw' folder.
    - file_type: csv, excel, or parquet
    """
    base_path = os.path.join(os.path.dirname(__file__), "../../data/raw")
    full_path = os.path.abspath(os.path.join(base_path, relative_path))
    return DataLoader(full_path).load(file_type=file_type, **kwargs)
