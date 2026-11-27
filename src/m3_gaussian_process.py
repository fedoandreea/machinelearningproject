import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from data_loading import load_case1
from evaluation import rmse, mae


def group_k_fold(groups, n_splits=5, seed=0):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    for val_groups in np.array_split(unique_groups, n_splits):
        is_val = np.isin(groups, val_groups)
        yield np.where(~is_val)[0], np.where(is_val)[0]


def make_kernel(kernel_name):
    local = ConstantKernel(1.0, (1e-3, 1e5)) * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-3, 1e5), nu=2.5)
    smooth = ConstantKernel(1.0, (1e-3, 1e5)) * RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-3, 1e5))
    trend = ConstantKernel(0.1, (1e-4, 1e3)) * RBF(length_scale=[3.0, 3.0, 3.0], length_scale_bounds=(1e-2, 1e5))
    noise = WhiteKernel(1e-3, (1e-8, 1e1))
    if kernel_name == "rbf":
        return smooth + noise
    if kernel_name == "matern":
        return local + noise
    return local + trend + noise


class PumpGaussianProcess:
    def __init__(self, feature_set="affinity", kernel_name="multiscale"):
        self.feature_set = feature_set
        self.kernel_name = kernel_name

    def features(self, data):
        n = data["speed_krpm"].to_numpy()
        power = data["power_w"].to_numpy()
        pressure_rise = data["pressure_rise_bar"].to_numpy()
        if self.feature_set == "affinity":
            return np.column_stack([pressure_rise / n ** 2, power / n ** 3, n])
        return np.column_stack([n, power, pressure_rise])

    def target(self, data):
        flow = data["flow_m3_h"].to_numpy()
        if self.feature_set == "affinity":
            return flow / data["speed_krpm"].to_numpy()
        return flow

    def fit(self, data):
        X = self.features(data)
        self.scaler = StandardScaler().fit(X)
        self.gp = GaussianProcessRegressor(
            kernel=make_kernel(self.kernel_name),
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=0,
        )
        self.gp.fit(self.scaler.transform(X), self.target(data))
        return self

    def predict(self, data):
        mean, std = self.gp.predict(self.scaler.transform(self.features(data)), return_std=True)
        if self.feature_set == "affinity":
            n = data["speed_krpm"].to_numpy()
            return mean * n, std * n
        return mean, std


train_data, test_data = load_case1()
y_test = test_data["flow_m3_h"].to_numpy()

groups = (train_data["pump_id"].astype(str) + "_" + (train_data["elapsed_s"] // 300).astype(str)).to_numpy()
folds = list(group_k_fold(groups))

best_score = np.inf
best_setup = None
for feature_set in ["raw", "affinity"]:
    for kernel_name in ["rbf", "matern", "multiscale"]:
        scores = []
        for train_idx, val_idx in folds:
            fold_train = train_data.iloc[train_idx]
            fold_val = train_data.iloc[val_idx]
            model = PumpGaussianProcess(feature_set, kernel_name).fit(fold_train)
            prediction, _ = model.predict(fold_val)
            scores.append(rmse(fold_val["flow_m3_h"].to_numpy(), prediction))
        score = np.mean(scores)
        print(feature_set, kernel_name, "CV RMSE:", round(score, 4))
        if score < best_score:
            best_score = score
            best_setup = (feature_set, kernel_name)

print("\nbest setup:", best_setup, "| CV RMSE:", best_score)

model = PumpGaussianProcess(*best_setup).fit(train_data)
predictions, prediction_std = model.predict(test_data)

print("learned kernel:", model.gp.kernel_)
print("log marginal likelihood:", model.gp.log_marginal_likelihood_value_)
print("\ntest RMSE:", rmse(y_test, predictions))
print("test MAE:", mae(y_test, predictions))
print("test R²:", 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
print("95% interval coverage:", np.mean(np.abs(y_test - predictions) <= 1.96 * prediction_std))

os.makedirs("figures", exist_ok=True)
errors = predictions - y_test
plt.figure(figsize=(7, 4))
plt.errorbar(y_test, errors, yerr=1.96 * prediction_std, fmt="o", markersize=3, alpha=0.6, capsize=2)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("true flow Q [m³/h]")
plt.ylabel("prediction error [m³/h]")
plt.title("M3 Gaussian Process: error vs flow (with 95% interval)")
plt.tight_layout()
plt.savefig("figures/m3_error_vs_flow.png", dpi=200)
plt.show()