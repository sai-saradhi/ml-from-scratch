import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(data):
    data["Pass"] = (data["GPA"] >= 2.0).astype(int)
    
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)

    return -np.mean(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )


def compute_gradients(X, y, y_pred):
    m = len(X)

    dW = (1 / m) * X.T @ (y_pred - y)
    db = (1 / m) * np.sum(y_pred - y)

    return dW, db


def train_logistic_regression(X, y, epochs=5000, lr=0.1):
    W, b = initialize_parameters(X.shape[1])
    losses = []
    m = len(X)

    for epoch in range(epochs):
        z = X @ W + b
        y_pred = sigmoid(z)

        dW = (1 / m) * X.T @ (y_pred - y)
        db = (1 / m) * np.sum(y_pred - y)

        W = W - lr * dW
        b = b - lr * db

        bce = compute_loss(y, y_pred)
        losses.append(bce)

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: Loss = {bce}")

    return W, b, losses


def acc(y, p):
    return np.mean(y == p)


np.random.seed(42)

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

y = data[["Pass"]].values

X_train, X_test, y_train, y_test = train_test_split_numpy(X, y)

X_train, X_test = scale_features(
    X_train,
    X_test
)

W, b, losses = train_logistic_regression(
    X_train,
    y_train
)

y_pred = sigmoid(X_test @ W + b)

y_pred_class = (y_pred >= 0.5).astype(int)


print("W =", W)
print("b =", b)
print("Accuracy =", acc(y_test, y_pred_class))

