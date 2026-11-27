import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_loading import load_case1
from evaluation import rmse, mae

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def group_k_fold(groups, n_splits=5, seed=0):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    for val_groups in np.array_split(unique_groups, n_splits):
        is_val = np.isin(groups, val_groups)
        yield np.where(~is_val)[0], np.where(is_val)[0]


def affinity_features(data):
    n = data["speed_krpm"].to_numpy()
    return np.column_stack([data["pressure_rise_bar"].to_numpy() / n ** 2, data["power_w"].to_numpy() / n ** 3, n])


def nn_ensemble_predict(train, test, n_models=5):
    X_train, X_test = affinity_features(train), affinity_features(test)
    scaler = StandardScaler().fit(X_train)
    y = train["flow_m3_h"].to_numpy() / train["speed_krpm"].to_numpy()
    y_mean, y_std = y.mean(), y.std()
    predictions = []
    for seed in range(n_models):
        net = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", alpha=1e-4, learning_rate_init=1e-3,
                           batch_size=64, max_iter=1500, tol=0.0, n_iter_no_change=1500, random_state=seed)
        net.fit(scaler.transform(X_train), (y - y_mean) / y_std)
        predictions.append(net.predict(scaler.transform(X_test)) * y_std + y_mean)
    return np.mean(predictions, axis=0) * test["speed_krpm"].to_numpy()


def gp_predict(train, test):
    X_train, X_test = affinity_features(train), affinity_features(test)
    scaler = StandardScaler().fit(X_train)
    kernel = (ConstantKernel(1.0, (1e-3, 1e5)) * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-3, 1e5), nu=2.5)
              + ConstantKernel(0.1, (1e-4, 1e3)) * RBF(length_scale=[3.0, 3.0, 3.0], length_scale_bounds=(1e-2, 1e5))
              + WhiteKernel(1e-3, (1e-8, 1e1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    gp.fit(scaler.transform(X_train), train["flow_m3_h"].to_numpy() / train["speed_krpm"].to_numpy())
    return gp.predict(scaler.transform(X_test)) * test["speed_krpm"].to_numpy()


train_data, test_data = load_case1()
y_test = test_data["flow_m3_h"].to_numpy()

groups = (train_data["pump_id"].astype(str) + "_" + (train_data["elapsed_s"] // 300).astype(str)).to_numpy()
folds = list(group_k_fold(groups))

weights = [0.0, 0.25, 0.5, 0.75, 1.0]
scores = {"nn ensemble": [], "gp": [], **{f"blend w_gp={w}": [] for w in weights}}
for train_idx, val_idx in folds:
    fold_train, fold_val = train_data.iloc[train_idx], train_data.iloc[val_idx]
    y_val = fold_val["flow_m3_h"].to_numpy()
    nn_pred = nn_ensemble_predict(fold_train, fold_val)
    gp_pred = gp_predict(fold_train, fold_val)
    scores["nn ensemble"].append(rmse(y_val, nn_pred))
    scores["gp"].append(rmse(y_val, gp_pred))
    for w in weights:
        scores[f"blend w_gp={w}"].append(rmse(y_val, w * gp_pred + (1 - w) * nn_pred))

for name, values in scores.items():
    print(f"{name:<18} CV RMSE: {np.mean(values):.4f}")

best_w = min(weights, key=lambda w: np.mean(scores[f"blend w_gp={w}"]))
print("\nbest blend weight for GP:", best_w)

nn_test = nn_ensemble_predict(train_data, test_data)
gp_test = gp_predict(train_data, test_data)
blend_test = best_w * gp_test + (1 - best_w) * nn_test
for name, prediction in [("nn ensemble", nn_test), ("gp", gp_test), ("blend", blend_test)]:
    r2 = 1 - np.sum((y_test - prediction) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    print(f"{name:<12} test RMSE: {rmse(y_test, prediction):.4f} | MAE: {mae(y_test, prediction):.4f} | R²: {r2:.5f}")

print()
for name, prediction in [("nn ensemble", nn_test), ("gp", gp_test), ("blend", blend_test)]:
    errors = prediction - y_test
    low = y_test < 5
    fold_scores = scores[name if name != "blend" else f"blend w_gp={best_w}"]
    print(f"{name:<12} max |err|: {np.max(np.abs(errors)):.3f} | 95th pct |err|: {np.percentile(np.abs(errors), 95):.3f}"
          f" | bias: {errors.mean():+.3f} | low-flow (<5) MAE: {np.mean(np.abs(errors[low])):.3f}"
          f" | CV fold std: {np.std(fold_scores):.3f}")