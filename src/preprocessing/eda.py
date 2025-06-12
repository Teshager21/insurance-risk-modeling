import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

sns.set(style="whitegrid")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class EDAUtils:
    """
    A utility class for performing Exploratory Data Analysis (EDA).
    Includes methods for univariate analysis.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.df = df

    def univariate_analysis(
        self, save_dir: Optional[str] = None, bins: int = 30, top_n: int = 20
    ):
        """
        Perform univariate analysis (distribution plots) on all features.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save the plots. If None, plots are displayed inline.
        bins : int
            Number of bins for histograms of numeric variables.
        top_n : int
            Number of top categories to show in bar plots for categorical features.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        logger.info("Starting univariate analysis...")

        for col in self.df.columns:
            plt.figure(figsize=(8, 4))
            try:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    sns.histplot(
                        self.df[col].dropna(), kde=True, bins=bins, color="steelblue"
                    )
                    plt.title(f"Distribution of {col}")
                elif (
                    pd.api.types.is_categorical_dtype(self.df[col])
                    or self.df[col].dtype == "object"
                ):
                    value_counts = self.df[col].value_counts().head(top_n)
                    sns.barplot(
                        x=value_counts.values, y=value_counts.index, palette="muted"
                    )
                    plt.title(f"Frequency of {col} (top {top_n} categories)")
                else:
                    logger.warning(f"Skipping unsupported column type: {col}")
                    continue

                plt.tight_layout()

                if save_dir:
                    path = os.path.join(save_dir, f"{col}_univariate.png")
                    plt.savefig(path)
                    logger.info(f"Saved plot: {path}")
                else:
                    plt.show()

            except Exception as e:
                logger.error(f"Failed to plot column {col}: {e}", exc_info=True)
            finally:
                plt.close()

        logger.info("Univariate analysis completed.")


class MultivariateAnalysis:
    """
    Multivariate EDA: Includes pair plots, correlation heatmaps, and PCA visualization.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = df.copy()
        sns.set_context("talk")  # Larger font size
        sns.set_style("white")  # Cleaner background

    def pairplot(
        self,
        features: Optional[List[str]] = None,
        hue: Optional[str] = None,
        diag_kind: str = "kde",
        palette: str = "Set2",
        save_path: Optional[str] = None,
    ) -> None:
        """Generate a large, full-width pair plot for selected features."""
        if features is None:
            features = self.df.select_dtypes(include="number").columns.tolist()

        plot_df = self.df[features].copy()
        if hue:
            plot_df[hue] = self.df[hue]

        pair_grid = sns.pairplot(
            plot_df,
            hue=hue,
            diag_kind=diag_kind,
            palette=palette,
            corner=True,
        )
        pair_grid.fig.set_size_inches(16, 12)
        pair_grid.fig.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    def correlation_heatmap(
        self,
        features: Optional[List[str]] = None,
        cmap: str = "coolwarm",
        figsize: tuple = (16, 10),
        annot: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Full-screen correlation heatmap."""
        if features is None:
            features = self.df.select_dtypes(include="number").columns.tolist()

        corr = self.df[features].corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation Heatmap", fontsize=18)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    def pca_analysis(
        self,
        features: Optional[List[str]] = None,
        hue: Optional[str] = None,
        n_components: int = 2,
        scale: bool = True,
        palette: str = "Set2",
        save_path: Optional[str] = None,
    ) -> None:
        """Large PCA scatter plot with PC1 and PC2 axes."""
        if features is None:
            features = self.df.select_dtypes(include="number").columns.tolist()

        data = self.df[features].dropna()
        labels = self.df[hue].loc[data.index] if hue else None

        data_scaled = StandardScaler().fit_transform(data) if scale else data.values
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data_scaled)

        pca_df = pd.DataFrame(
            components,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=data.index,
        )
        if hue:
            pca_df[hue] = labels

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=hue,
            palette=palette,
            s=80,
            alpha=0.8,
            edgecolor="k",
        )
        plt.title("PCA: First Two Principal Components", fontsize=18)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
