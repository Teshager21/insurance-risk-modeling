# target_encoder_wrapper.py

from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder


class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible wrapper for category_encoders.TargetEncoder.
    Allows use within ColumnTransformer by properly handling target during fit.
    """

    def __init__(self):
        self.encoder = TargetEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def fit_transform(self, X, y=None):
        return self.encoder.fit_transform(X, y)
