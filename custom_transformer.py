from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def log_transform(X):
    return np.log1p(X)

def square_transform(X):
    return X ** 2

def log_square_transform(X):
    return np.log1p(X) ** 2

preprocessor = ColumnTransformer(
    transformers=[
        ('log_sensing_score', FunctionTransformer(log_transform), ['Sensing Score']),
        ('log_judging_score', FunctionTransformer(log_transform), ['Judging Score']),
        ('sensing_score_squared', FunctionTransformer(square_transform), ['Sensing Score']),
        ('judging_score_squared', FunctionTransformer(square_transform), ['Judging Score']),
        ('log_age_squared', FunctionTransformer(log_square_transform), ['Age'])
    ],
    remainder='passthrough'
)