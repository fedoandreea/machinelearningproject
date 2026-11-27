import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loading import load_case1
from evaluation import rmse, mae
import os
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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

neural_network = TransformedTargetRegressor(
    regressor=Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            solver="adam",
            batch_size=64,
            max_iter=1500,
            tol=0.0,
            n_iter_no_change=1500,
            random_state=0,
        )),
    ]),
    transformer=StandardScaler(),
)

parameter_grid = {
    "regressor__mlp__hidden_layer_sizes": [(16,), (32,), (32, 32), (64, 64)],
    "regressor__mlp__activation": ["tanh", "relu"],
    "regressor__mlp__alpha": [1e-4, 1e-2],
    "regressor__mlp__learning_rate_init": [1e-3, 1e-2],
}

search = GridSearchCV(
    neural_network,
    parameter_grid,
    scoring="neg_root_mean_squared_error",
    cv=list(group_k_fold(cv_groups)),
    n_jobs=-1,
)
search.fit(X_train, y_train)

print("best hyperparameters:", search.best_params_)
print("best CV RMSE:", -search.best_score_)

predictions = search.best_estimator_.predict(X_test)
print("test RMSE:", rmse(y_test, predictions))
print("test MAE:", mae(y_test, predictions))