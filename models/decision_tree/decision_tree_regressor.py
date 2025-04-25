from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


def relative_error_criterion(y_left, y_right, feature_idx=None, threshold=None, rho_star_idx=None):
    def compute_relative_error(y):
        pred = np.mean(y)
        rel_error = np.abs(y - pred) / np.maximum(np.abs(y), 1e-8)
        return np.mean(rel_error)

    left_error = compute_relative_error(y_left)
    right_error = compute_relative_error(y_right)

    total_len = len(y_left) + len(y_right)
    weighted_error = (len(y_left) * left_error + len(y_right) * right_error) / total_len

    # Apply weight of 8 if splitting on rho* feature at threshold 0
    if (feature_idx is not None and rho_star_idx is not None and
            feature_idx == rho_star_idx and abs(threshold) < 1e-8):
        weighted_error *= 8

    return weighted_error * 100


class CustomDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_depth=10, min_samples_split=2, criterion=relative_error_criterion, rho_star_idx=None):
        self.tree_ = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.rho_star_idx = rho_star_idx  # Index of the rho* feature

    def fit(self, X, y):
        self.tree_ = self._build_tree(np.array(X), np.array(y), depth=0)
        return self

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        best_score = float('inf')
        best_split = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = X[:, feature] <= t
                right = ~left
                if len(y[left]) == 0 or len(y[right]) == 0:
                    continue
                # Pass the feature index and threshold to the criterion
                score = self.criterion(y[left], y[right],
                                       feature_idx=feature,
                                       threshold=t,
                                       rho_star_idx=self.rho_star_idx)
                if score < best_score:
                    best_score = score
                    best_split = {
                        'feature': feature,
                        'threshold': t,
                        'left_X': X[left],
                        'left_y': y[left],
                        'right_X': X[right],
                        'right_y': y[right]
                    }

        if best_split is None:
            return np.mean(y)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': self._build_tree(best_split['left_X'], best_split['left_y'], depth + 1),
            'right': self._build_tree(best_split['right_X'], best_split['right_y'], depth + 1)
        }

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x, self.tree_) for x in X])