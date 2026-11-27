import numpy as np


class NeuralNetworkScratch:
    def __init__(self, hidden_layers=(32,), activation="tanh", alpha=1e-4,
                 learning_rate=1e-3, epochs=1500, batch_size=64, seed=0):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def activation_function(self, z):
        if self.activation == "tanh":
            return np.tanh(z)
        return np.maximum(0, z)

    def activation_derivative(self, a):
        if self.activation == "tanh":
            return 1 - a ** 2
        return (a > 0) * 1.0

    def init_weights(self, n_inputs):
        sizes = [n_inputs] + list(self.hidden_layers) + [1]
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            limit = np.sqrt(6 / (sizes[i] + sizes[i + 1]))
            self.W.append(self.rng.uniform(-limit, limit, size=(sizes[i], sizes[i + 1])))
            self.b.append(self.rng.uniform(-limit, limit, size=sizes[i + 1]))

    def forward(self, X):
        layers = [X]
        for i in range(len(self.W)):
            z = layers[-1] @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                z = self.activation_function(z)
            layers.append(z)
        return layers

    def loss(self, X, y):
        prediction = self.forward(X)[-1]
        penalty = sum(np.sum(w ** 2) for w in self.W)
        return 0.5 * np.mean((prediction - y) ** 2) + 0.5 * self.alpha * penalty / len(X)

    def backward(self, X, y):
        layers = self.forward(X)
        n = len(X)
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        delta = (layers[-1] - y) / n
        for i in range(len(self.W) - 1, -1, -1):
            dW[i] = layers[i].T @ delta + self.alpha * self.W[i] / n
            db[i] = np.sum(delta, axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.activation_derivative(layers[i])
        return dW, db

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.y_mean = y.mean()
        self.y_std = y.std()
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        self.rng = np.random.default_rng(self.seed)
        self.init_weights(X.shape[1])

        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        self.loss_history = []
        for epoch in range(self.epochs):
            indices = self.rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                batch = indices[start:start + self.batch_size]
                dW, db = self.backward(X[batch], y[batch])
                t += 1
                for i in range(len(self.W)):
                    mW[i] = beta1 * mW[i] + (1 - beta1) * dW[i]
                    vW[i] = beta2 * vW[i] + (1 - beta2) * dW[i] ** 2
                    mb[i] = beta1 * mb[i] + (1 - beta1) * db[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] ** 2

                    self.W[i] -= self.learning_rate * (mW[i] / (1 - beta1 ** t)) / (np.sqrt(vW[i] / (1 - beta2 ** t)) + eps)
                    self.b[i] -= self.learning_rate * (mb[i] / (1 - beta1 ** t)) / (np.sqrt(vb[i] / (1 - beta2 ** t)) + eps)
            self.loss_history.append(self.loss(X, y))
        return self

    def predict(self, X):
        X = (np.array(X, dtype=float) - self.X_mean) / self.X_std
        prediction = self.forward(X)[-1]
        return (prediction * self.y_std + self.y_mean).ravel()


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def group_k_fold(groups, n_splits=5, seed=0):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    for val_groups in np.array_split(unique_groups, n_splits):
        is_val = np.isin(groups, val_groups)
        yield np.where(~is_val)[0], np.where(is_val)[0]


def gradient_check(activation):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=(10, 1))

    net = NeuralNetworkScratch(hidden_layers=(5, 4), activation=activation, alpha=0.01)
    net.rng = rng
    net.init_weights(3)
    dW, db = net.backward(X, y)

    eps = 1e-6
    biggest_error = 0
    for params, grads in [(net.W, dW), (net.b, db)]:
        for p, g in zip(params, grads):
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + eps
                loss_plus = net.loss(X, y)
                p[idx] = old - eps
                loss_minus = net.loss(X, y)
                p[idx] = old
                numeric = (loss_plus - loss_minus) / (2 * eps)
                error = abs(numeric - g[idx]) / max(1e-8, abs(numeric) + abs(g[idx]))
                biggest_error = max(biggest_error, error)
    return biggest_error


def cross_validation(X, y, groups):
    folds = list(group_k_fold(groups))
    best_score = np.inf
    best_params = None

    for hidden in [(16,), (32,), (32, 32), (64, 64)]:
        for activation in ["tanh", "relu"]:
            for alpha in [1e-4, 1e-2]:
                for lr in [1e-3, 1e-2]:
                    scores = []
                    for train_idx, val_idx in folds:
                        net = NeuralNetworkScratch(hidden, activation, alpha, lr)
                        net.fit(X[train_idx], y[train_idx])
                        scores.append(rmse(y[val_idx], net.predict(X[val_idx])))
                    score = np.mean(scores)
                    print(hidden, activation, alpha, lr, "CV RMSE:", round(score, 4))

                    if score < best_score:
                        best_score = score
                        best_params = {"hidden_layers": hidden, "activation": activation,
                                       "alpha": alpha, "learning_rate": lr}

    return best_params, best_score


if __name__ == "__main__":
    from data_loading import load_case1

    print("gradient check tanh:", gradient_check("tanh"))
    print("gradient check relu:", gradient_check("relu"))

    train_data, test_data = load_case1()
    features = ["speed_krpm", "power_w", "pressure_rise_bar"]

    X_train = train_data[features].to_numpy()
    y_train = train_data["flow_m3_h"].to_numpy()
    X_test = test_data[features].to_numpy()
    y_test = test_data["flow_m3_h"].to_numpy()

    groups = (train_data["pump_id"].astype(str) + "_" + (train_data["elapsed_s"] // 300).astype(str)).to_numpy()

    best_params, best_score = cross_validation(X_train, y_train, groups)
    print("\nbest hyperparameters:", best_params)
    print("best CV RMSE:", best_score)

    net = NeuralNetworkScratch(**best_params).fit(X_train, y_train)
    predictions = net.predict(X_test)
    print("\nscratch   test RMSE:", rmse(y_test, predictions), "| MAE:", mae(y_test, predictions))

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    reference = TransformedTargetRegressor(
        regressor=Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=best_params["hidden_layers"],
                activation=best_params["activation"],
                alpha=best_params["alpha"],
                learning_rate_init=best_params["learning_rate"],
                solver="adam", batch_size=64, max_iter=1500, tol=0.0, n_iter_no_change=1500, random_state=0,
            )),
        ]),
        transformer=StandardScaler(),
    ).fit(X_train, y_train)
    reference_predictions = reference.predict(X_test)
    print("reference test RMSE:", rmse(y_test, reference_predictions), "| MAE:", mae(y_test, reference_predictions))