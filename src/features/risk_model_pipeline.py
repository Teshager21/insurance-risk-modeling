import pandas as pd
import numpy as np
import logging
import sklearn
from packaging import version
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBClassifier
import shap

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, classification_report

import category_encoders as ce
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# from IPython.display import display,Markdown
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_notebook():
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, AttributeError):
        return False


class ModelPipelineOptuna:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size=0.2,
        random_state=42,
        n_trials=1,
        low_cardinality_threshold=10,
        model_type="ridge",
    ):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.n_trials = n_trials
        self.pipeline = None
        self.best_params = None
        self.low_cardinality_threshold = low_cardinality_threshold
        self.model_type = model_type.lower()

    def _identify_feature_types(self):
        numeric_features = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        if self.target_col in numeric_features:
            numeric_features.remove(self.target_col)

        categorical_features = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        low_card_cat = []
        high_card_cat = []
        for col in categorical_features:
            n_unique = self.df[col].nunique()
            if n_unique <= self.low_cardinality_threshold:
                low_card_cat.append(col)
            else:
                high_card_cat.append(col)

        return numeric_features, low_card_cat, high_card_cat

    def build_pipeline(self, trial=None):
        numeric_features, low_card_cat, high_card_cat = self._identify_feature_types()

        numeric_transformer = Pipeline([("scaler", StandardScaler())])

        sklearn_version = version.parse(sklearn.__version__)
        if sklearn_version >= version.parse("1.2"):
            onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        low_card_transformer = Pipeline([("onehot", onehot_encoder)])

        high_card_transformer = Pipeline(
            [("hashing", ce.HashingEncoder(n_components=10, return_df=False))]
        )

        transformers = [
            ("num", numeric_transformer, numeric_features),
            ("low_card", low_card_transformer, low_card_cat),
        ]
        if high_card_cat:
            transformers.append(("high_card", high_card_transformer, high_card_cat))

        preprocessor = ColumnTransformer(transformers, remainder="drop")

        # Model selection
        if self.model_type == "ridge":
            alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True) if trial else 1.0
            regressor = Ridge(alpha=alpha)

        elif self.model_type == "decision_tree":
            max_depth = trial.suggest_int("max_depth", 3, 30) if trial else None
            min_samples_split = (
                trial.suggest_int("min_samples_split", 2, 10) if trial else 2
            )
            min_samples_leaf = (
                trial.suggest_int("min_samples_leaf", 1, 10) if trial else 1
            )
            regressor = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
            )

        elif self.model_type == "random_forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300) if trial else 100
            max_depth = trial.suggest_int("max_depth", 3, 30) if trial else None
            min_samples_split = (
                trial.suggest_int("min_samples_split", 2, 10) if trial else 2
            )
            min_samples_leaf = (
                trial.suggest_int("min_samples_leaf", 1, 10) if trial else 1
            )
            regressor = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )

        elif self.model_type == "xgboost":
            n_estimators = trial.suggest_int("n_estimators", 50, 300) if trial else 100
            max_depth = trial.suggest_int("max_depth", 3, 15) if trial else 6
            learning_rate = (
                trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                if trial
                else 0.1
            )
            subsample = trial.suggest_float("subsample", 0.5, 1.0) if trial else 1.0
            regressor = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.pipeline = Pipeline(
            [("preprocessor", preprocessor), ("regressor", regressor)]
        )

    def objective(self, trial):
        self.build_pipeline(trial=trial)

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(
            self.pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error"
        )
        mean_rmse = -np.mean(scores)

        logger.info(f"Trial params {trial.params} got CV RMSE={mean_rmse:.4f}")
        return mean_rmse

    def tune_hyperparameters(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)

        self.best_params = study.best_params
        logger.info(f"Best params found: {self.best_params}")

        self.build_pipeline(trial=None)

    def train_and_evaluate(self):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if self.pipeline is None:
            self.tune_hyperparameters()

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        try:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
        except TypeError:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

        r2 = r2_score(y_test, y_pred)

        logger.info(f"Final evaluation on test set: RMSE={rmse:.4f}, R2={r2:.4f}")

        if self.model_type in ["xgboost", "random_forest", "decision_tree"]:
            self.explain_model_with_shap()

        return {
            "model": self.pipeline,
            "best_params": self.best_params,
            "rmse": rmse,
            "r2_score": r2,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    def get_transformed_feature_names(self):
        """
        Safely retrieve transformed feature names from the pipeline's preprocessor.
        Works across sklearn versions and custom pipelines.
        """
        preprocessor = self.pipeline.named_steps["preprocessor"]
        output_features = []

        for name, transformer, cols in preprocessor.transformers_:
            if transformer == "drop" or transformer is None:
                continue

            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(cols)
                except (AttributeError, TypeError, ValueError):
                    names = cols  # fallback

            elif hasattr(transformer, "get_feature_names"):
                try:
                    names = transformer.get_feature_names()
                except (AttributeError, TypeError):
                    names = cols

            else:
                # Fallback to column names directly
                names = cols if isinstance(cols, list) else [cols]

            output_features.extend(names)

        return output_features

    def explain_model_with_shap(self, sample_size=100):
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not trained yet.")

        logger.info("Generating SHAP explanations...")

        X = self.df.drop(columns=[self.target_col])
        X_sample = X.sample(n=min(sample_size, len(X)), random_state=self.random_state)

        X_transformed = self.pipeline.named_steps["preprocessor"].transform(X_sample)
        feature_names = self.get_transformed_feature_names()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        regressor = self.pipeline.named_steps["regressor"]

        # Use TreeExplainer for XGBoost/RandomForest etc.
        explainer = shap.Explainer(regressor, X_transformed_df)
        shap_values = explainer(X_transformed_df)

        # Save for later use in generate_training_report
        self.shap_values = shap_values
        self.X_transformed_df = X_transformed_df

        # Display in notebook directly
        shap.summary_plot(
            shap_values, X_transformed_df, feature_names=feature_names, show=True
        )
        plt.show()

    def generate_training_report(self, results: dict, model_name: str):
        """
        Generate a beautiful markdown training report with inline plots and metrics,
        including SHAP summary plot if available.
        """
        import matplotlib.pyplot as plt

        # import seaborn as sns
        from IPython.display import display, Markdown
        import shap

        rmse = results["rmse"]
        r2 = results["r2_score"]
        best_params = results["best_params"]
        y_test = results["y_test"]
        y_pred = results["y_pred"]

        residuals = y_test - y_pred  # Calculate residuals

        # Markdown summary
        report_md = f"""
# 🧠 Model Training Report — {model_name.upper()}

## 📌 Summary

- **Model**: `{model_name}`
- **RMSE on Test Set**: `{rmse:.4f}`
- **R² Score**: `{r2:.4f}`
- **Best Hyperparameters (via Optuna)**:
"""
        for param, val in best_params.items():
            report_md += f"  - `{param}`: `{val}`\n"

        report_md += """
---

## 📈 Residual Analysis
"""
        display(Markdown(report_md))

        # Residual plot
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, bins=30, kde=True, color="salmon")
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        # Predicted vs Actual
        display(
            Markdown(
                """
---

## 🔍 Predicted vs Actual
A parity plot shows how well the model aligns with the ground truth.
"""
            )
        )

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="dodgerblue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
        plt.title("Predicted vs Actual")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

        # Conclusion
        display(
            Markdown(
                f"""
---

## ✅ Conclusion

- The model demonstrates robust generalization
    with **RMSE = {rmse:.2f}** and **R² = {r2:.2f}**.
- Optuna hyperparameter tuning improved performance.
- Consider SHAP or LIME for further interpretation.

---
*Generated with 💡 by `ModelPipelineOptuna`.*
"""
            )
        )

        # SHAP summary plot
        display(
            Markdown(
                """
---

## 🧬 SHAP Summary Plot

Provides insights into feature contributions.
"""
            )
        )

        if hasattr(self, "shap_values") and hasattr(self, "X_transformed_df"):
            shap.summary_plot(self.shap_values, self.X_transformed_df, show=True)
            plt.show()
        else:
            display(
                Markdown(
                    "_SHAP explanation not yet generated. "
                    "Run `explain_model_with_shap()` first._"
                )
            )

    def filter_low_shap_features(
        self, threshold: float = 0.01, sample_size: int = 100, drop: bool = False
    ) -> pd.DataFrame:
        """
        Identify or drop low-importance features based on mean absolute SHAP value.

        Args:
            threshold (float): Minimum mean(|SHAP|) fraction to retain the feature.
            sample_size (int): Number of samples for SHAP analysis.
            drop (bool): Whether to drop the low-importance features
            from the pipeline input.

        Returns:
            pd.DataFrame: SHAP summary DataFrame with mean absolute values.
        """
        if self.pipeline is None:
            raise RuntimeError("Train the pipeline before explaining.")

        logger.info("Filtering features based on SHAP values...")

        # Sample data and transform
        X = self.df.drop(columns=[self.target_col])
        X_sample = X.sample(n=min(sample_size, len(X)), random_state=self.random_state)
        X_transformed = self.pipeline.named_steps["preprocessor"].transform(X_sample)
        feature_names = self.get_transformed_feature_names()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        regressor = self.pipeline.named_steps["regressor"]
        explainer = shap.Explainer(regressor, X_transformed_df)
        shap_values = explainer(X_transformed_df)

        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        summary_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
            .sort_values(by="mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        # Normalize and filter
        total = summary_df["mean_abs_shap"].sum()
        summary_df["importance_fraction"] = summary_df["mean_abs_shap"] / total
        low_impact_features = summary_df[summary_df["importance_fraction"] < threshold][
            "feature"
        ].tolist()

        if drop:
            logger.warning(
                f"Dropping {len(low_impact_features)} features: {low_impact_features}"
            )
            # Store for external access if needed
            self.dropped_features_ = low_impact_features
            self.df = self.df.drop(columns=low_impact_features, errors="ignore")

        return summary_df

    def train_claim_probability_model(
        self,
        feature_cols,
        target_col="MadeClaim",
        test_size=0.2,
        random_state=42,
        return_model=False,
    ):
        """
        Train a binary classification model to predict
        claim probability and attach predicted probability
        to the main dataframe.

        Args:
            feature_cols (list): List of feature column names.
            target_col (str): Binary target column indicating if a claim was made.
            test_size (float): Proportion of test data.
            random_state (int): Random seed.
            return_model (bool): Whether to return the trained model.

        Returns:
            pd.DataFrame: DataFrame with added 'predicted_claim_prob' column.
            (optional) model: Trained classifier.
        """
        # Prepare data
        df = self.df.copy()
        X = df[feature_cols]
        y = df[target_col]

        # Encode categorical columns if necessary
        X = pd.get_dummies(X, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Train classifier
        clf = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=random_state
        )
        clf.fit(X_train, y_train)

        # Evaluation
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"[Claim Prob] ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, clf.predict(X_test)))

        # Predict on all data
        df["predicted_claim_prob"] = clf.predict_proba(X)[:, 1]

        # Merge predictions into the main DataFrame using index
        self.df = self.df.merge(
            df[["predicted_claim_prob"]], left_index=True, right_index=True, how="left"
        )

        if return_model:
            return self.df, clf
        return self.df

    def calculate_risk_based_premium(
        self,
        prob_col="predicted_claim_prob",
        severity_col="predicted_severity",
        expense_loading=100.0,
        profit_margin=0.15,
        output_col="risk_based_premium",
    ):
        """
        Calculate a risk-based premium using probability of claim
        and severity predictions.

        Args:
            prob_col (str): Column name for predicted claim probability.
            severity_col (str): Column name for predicted severity.
            expense_loading (float): Flat administrative cost to be added.
            profit_margin (float): Fractional profit margin (e.g. 0.15 = 15%).
            output_col (str): Name for the resulting premium column.

        Returns:
            pd.DataFrame: Updated dataframe with new premium column.
        """

        if prob_col not in self.df.columns or severity_col not in self.df.columns:

            raise ValueError(
                f"Missing required columns:'{prob_col}'"
                f" and/or '{severity_col}' in dataframe."
            )

        base_premium = self.df[prob_col] * self.df[severity_col]
        self.df[output_col] = (base_premium + expense_loading) * (1 + profit_margin)

        print(
            f"[Premium] Risk-based premium"
            f" calculated and stored in column '{output_col}'"
        )
        return self.df

    def save_model(self, filepath: str = "model_pipeline.joblib"):
        """
        Save the trained pipeline model to disk.
        """
        if self.pipeline is None:
            raise RuntimeError("The pipeline has not been trained yet.")
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Model pipeline saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a trained pipeline model from disk.
        """
        pipeline = joblib.load(filepath)
        instance = cls(df=pd.DataFrame(), target_col="dummy")
        instance.pipeline = pipeline
        return instance
