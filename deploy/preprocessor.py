from sklearn.base import TransformerMixin, BaseEstimator

import config


class DoNothing(TransformerMixin, BaseEstimator):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        return X


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

pca = PCA()
standard = StandardScaler()

numeric_transformer = Pipeline([
    ('pca', pca),
    ('standard_scailer', standard)
])

preprocess = ColumnTransformer([
    ('num', numeric_transformer, config.NUMERIC_COL),
])
