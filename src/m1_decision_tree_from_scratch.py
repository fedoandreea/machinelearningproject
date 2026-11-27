import numpy as np


class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.feature_, self.threshold_, self.left_, self.right_, self.value_, self.depth_ = [], [], [], [], [], []
        self._build(X, y, depth=0)

        self.feature_ = np.array(self.feature_, dtype=int)
        self.threshold_ = np.array(self.threshold_, dtype=float)
        self.left_ = np.array(self.left_, dtype=int)
        self.right_ = np.array(self.right_, dtype=int)
        self.value_ = np.array(self.value_, dtype=float)
        self.depth_ = np.array(self.depth_, dtype=int)
        return self

    def _new_leaf(self, value, depth):
        self.feature_.append(-1)
        self.threshold_.append(np.nan)
        self.left_.append(-1)
        self.right_.append(-1)
        self.value_.append(value)
        self.depth_.append(depth)
        return len(self.value_) - 1

    def _build(self, X, y, depth):
        node = self._new_leaf(y.mean(), depth)

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature, threshold = split
        goes_left = X[:, feature] <= threshold
        self.feature_[node] = feature
        self.threshold_[node] = threshold
        self.left_[node] = self._build(X[goes_left], y[goes_left], depth + 1)
        self.right_[node] = self._build(X[~goes_left], y[~goes_left], depth + 1)
        return node

    def _best_split(self, X, y):
        n_samples = len(y)
        y_centered = y - y.mean()
        if np.sum(y_centered**2) <= 1e-12:
            return None

        best_sse, best_split = np.inf, None
        left_count = np.arange(1, n_samples)
        right_count = n_samples - left_count

        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            x_sorted = X[order, feature]
            y_sorted = y_centered[order]

            cumulative_sum = np.cumsum(y_sorted)
            cumulative_sq = np.cumsum(y_sorted**2)
            left_sum, left_sq = cumulative_sum[:-1], cumulative_sq[:-1]
            right_sum = cumulative_sum[-1] - left_sum
            right_sq = cumulative_sq[-1] - left_sq
            children_sse = (left_sq - left_sum**2 / left_count) + (right_sq - right_sum**2 / right_count)

            valid = (
                (x_sorted[:-1] < x_sorted[1:])
                & (left_count >= self.min_samples_leaf)
                & (right_count >= self.min_samples_leaf)
            )
            if not valid.any():
                continue

            children_sse = np.where(valid, children_sse, np.inf)
            i = int(np.argmin(children_sse))
            if children_sse[i] < best_sse:
                threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
                if threshold == x_sorted[i + 1]:
                    threshold = x_sorted[i]
                best_sse, best_split = children_sse[i], (feature, threshold)

        return best_split

    def predict(self, X, depth_limit=None):
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        active = self._is_active(node, depth_limit)
        while active.any():
            idx = np.where(active)[0]
            current = node[idx]
            go_left = X[idx, self.feature_[current]] <= self.threshold_[current]
            node[idx] = np.where(go_left, self.left_[current], self.right_[current])
            active = self._is_active(node, depth_limit)
        return self.value_[node]

    def _is_active(self, node, depth_limit):
        active = self.feature_[node] >= 0
        if depth_limit is not None:
            active &= self.depth_[node] < depth_limit
        return active

    def get_n_leaves(self):
        return int(np.sum(self.feature_ == -1))


def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def group_k_fold(groups, n_splits=5, seed=0):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    for validation_groups in np.array_split(unique_groups, n_splits):
        is_validation = np.isin(groups, validation_groups)
        yield np.where(~is_validation)[0], np.where(is_validation)[0]


def tune_with_cross_validation(X, y, groups, depth_grid, leaf_grid, n_splits=5):
    deepest = max(depth_grid)
    cv_rmse = {}
    for min_samples_leaf in leaf_grid:
        fold_scores = {depth: [] for depth in depth_grid}
        for train_idx, validation_idx in group_k_fold(groups, n_splits):
            tree = DecisionTreeRegressorScratch(max_depth=deepest, min_samples_leaf=min_samples_leaf)
            tree.fit(X[train_idx], y[train_idx])
            for depth in depth_grid:
                predictions = tree.predict(X[validation_idx], depth_limit=depth)
                fold_scores[depth].append(root_mean_squared_error(y[validation_idx], predictions))
        for depth in depth_grid:
            cv_rmse[(depth, min_samples_leaf)] = float(np.mean(fold_scores[depth]))
            print(f"max_depth={depth:>2} min_samples_leaf={min_samples_leaf:>2} CV RMSE={cv_rmse[(depth, min_samples_leaf)]:.4f}")
    best_depth, best_leaf = min(cv_rmse, key=cv_rmse.get)
    return best_depth, best_leaf, cv_rmse


if __name__ == "__main__":
    from data_loading import load_case1

    train_data, test_data = load_case1()
    feature_columns = ["speed_krpm", "power_w", "pressure_rise_bar"]

    X_train = train_data[feature_columns].to_numpy()
    y_train = train_data["flow_m3_h"].to_numpy()
    X_test = test_data[feature_columns].to_numpy()
    y_test = test_data["flow_m3_h"].to_numpy()

    block_length_s = 300
    cv_groups = (
        train_data["pump_id"].astype(str) + "_" + (train_data["elapsed_s"] // block_length_s).astype(str)
    ).to_numpy()

    depth_grid = [2, 4, 6, 8, 10, 12, 15, 20]
    leaf_grid = [1, 5, 10, 20, 50]

    best_depth, best_leaf, cv_rmse = tune_with_cross_validation(X_train, y_train, cv_groups, depth_grid, leaf_grid)
    print("\nbest hyperparameters: max_depth =", best_depth, "| min_samples_leaf =", best_leaf)
    print("best CV RMSE:", cv_rmse[(best_depth, best_leaf)])

    final_tree = DecisionTreeRegressorScratch(max_depth=best_depth, min_samples_leaf=best_leaf).fit(X_train, y_train)
    predictions = final_tree.predict(X_test)
    print("\nscratch   test RMSE:", root_mean_squared_error(y_test, predictions),
          "| MAE:", mean_absolute_error(y_test, predictions), "| leaves:", final_tree.get_n_leaves())

    from sklearn.tree import DecisionTreeRegressor

    reference_tree = DecisionTreeRegressor(max_depth=best_depth, min_samples_leaf=best_leaf, random_state=0)
    reference_tree.fit(X_train, y_train)
    reference_predictions = reference_tree.predict(X_test)
    print("reference test RMSE:", root_mean_squared_error(y_test, reference_predictions),
          "| MAE:", mean_absolute_error(y_test, reference_predictions), "| leaves:", reference_tree.get_n_leaves())
    print("identical predictions:", np.mean(np.isclose(predictions, reference_predictions)) * 100, "%")
    train_identical = np.mean(np.isclose(final_tree.predict(X_train), reference_tree.predict(X_train))) * 100
print("identical TRAIN predictions:", train_identical, "%")

for seed in range(10):
    tree = DecisionTreeRegressor(max_depth=best_depth, min_samples_leaf=best_leaf, random_state=seed).fit(X_train, y_train)
    print("sklearn random_state", seed, "test RMSE:", root_mean_squared_error(y_test, tree.predict(X_test)))