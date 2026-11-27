import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from data_loading import load_case1
from evaluation import rmse, mae


def group_k_fold(groups, n_splits=5, seed=0):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    for validation_groups in np.array_split(unique_groups, n_splits):
        is_validation = np.isin(groups, validation_groups)
        yield np.where(~is_validation)[0], np.where(is_validation)[0]


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

parameter_grid = {
    "max_depth": [2, 4, 6, 8, 10, 12, 15, 20],
    "min_samples_leaf": [1, 5, 10, 20, 50],
}

search = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    parameter_grid,
    scoring="neg_root_mean_squared_error",
    cv=list(group_k_fold(cv_groups)),
    n_jobs=-1,
)
search.fit(X_train, y_train)

print("best hyperparameters:", search.best_params_)
print("best CV RMSE:", -search.best_score_)

best_tree = search.best_estimator_
print("tree depth:", best_tree.get_depth(), "| leaves:", best_tree.get_n_leaves())

predictions = best_tree.predict(X_test)
print("test RMSE:", rmse(y_test, predictions))
print("test MAE:", mae(y_test, predictions))