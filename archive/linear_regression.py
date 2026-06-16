import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(data):
    data["ParentalEducation"] = (
        data["ParentalEducation"]
        .map({
            "High School": 0,
            "Some College": 1,
            "Bachelor": 2,
            "Higher": 3
        })
        .fillna(4)
    )

    data["ParentalSupport"] = (
        data["ParentalSupport"]
        .map({
            "Low": 0,
            "Moderate": 1,
            "High": 2,
            "Very High": 3
        })
        .fillna(4)
    )

    data = pd.get_dummies(
        data,
        columns=["Ethnicity"],
        dtype=int
    )

    return data


def train_test_split_numpy(X, y, train_ratio=0.8):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def initialize_parameters(num_features):
    W = np.random.randn(num_features, 1)
    b = np.zeros((1, 1))

    return W, b


def compute_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2) 


def compute_gradients(X, y, y_pred):
    m = len(X)

    dW = (2 / m) * X.T @ (y_pred - y)
    db = (2 / m) * np.sum(y_pred - y)

    return dW, db


def train_linear_regression(X, y, epochs=35, lr=0.1):
    W, b = initialize_parameters(X.shape[1])
    losses = []

    for _ in range(epochs):
        y_pred = X @ W + b

        dW, db = compute_gradients(
            X,
            y,
            y_pred
        )

        W = W - lr * dW
        b = b - lr * db
        mse  = np.mean((y_pred - y) ** 2)
        losses.append(mse)

    return W, b, losses


def r2_score_numpy(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


np.random.seed(47)

data = pd.read_csv("data.csv")

data = preprocess_data(data)

X = data[
    [
        "StudyTimeWeekly",
        "Absences",
        "Age",
        "Tutoring",
        "Sports",
        "Extracurricular",
        "ParentalSupport",
        "Music"
    ]
].values

y = data[["GPA"]].values

X_train, X_test, y_train, y_test = train_test_split_numpy(X, y)

X_train, X_test = scale_features(
    X_train,
    X_test
)

W, b, losses = train_linear_regression(
    X_train,
    y_train
)

y_pred = X_test @ W + b

print("W =", W)
print("b =", b)
print("R2 =", r2_score_numpy(y_test, y_pred))
