from sklearn.pipeline import Pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import config
import preprocessor as pp


pipe = Pipeline([
    ('preprocessor', pp.preprocess),
    ('clf', LinearDiscriminantAnalysis())
])


# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer

# pca = PCA()
# standard = StandardScaler()

# numeric_transformer = Pipeline([
#     ('pca', pca),
#     ('standard_scailer', standard)
# ])

# preprocess = ColumnTransformer([
#     ('num', numeric_transformer, config.NUMERIC_COL),
# ])

# pipe = Pipeline([
#     ('preprocessor', preprocess),
#     ('clf', LinearDiscriminantAnalysis())
# ])
