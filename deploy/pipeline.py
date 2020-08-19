from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


pca = PCA()
standard = StandardScaler()

numeric_transformer = Pipeline([
    ('pca', pca),
    ('standard_scailer', standard)
])

numeric_cols = df.columns[:-1]

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
])
