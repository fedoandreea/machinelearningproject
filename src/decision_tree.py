"""M1: Decision Tree Regressor implemented from scratch (standard library, NumPy, SciPy only).

TODO (owner: ____):
- DecisionTreeRegressorScratch(max_depth, min_samples_split, min_samples_leaf)
- fit(X, y): recursively choose the (feature, threshold) split that minimises
  the weighted MSE of the two children; stop at max_depth / min samples.
- predict(X): walk each sample down the tree, return the leaf mean.
- Check correctness against sklearn.tree.DecisionTreeRegressor with the same hyperparameters.
"""
import numpy as np


class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
