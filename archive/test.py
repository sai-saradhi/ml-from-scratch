import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

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

def train_test_split(X, y, train_ratio = 0.8):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_index = len(X) * train_ratio
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    mean = X_train.mean(axis = 0)
    std = X_train.std(axis = 0)
    
    std[std == 0] = 1
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test

class LassoRegression:

    def __init__(self, alpha=0.1, lr=0.01, epochs=1000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs

    def _compute_loss(self, y_true, y_pred):
        mse = np.mean((y_pred - y_true) ** 2)
        l1 = self.alpha * np.sum(np.abs(self.W))
        return mse + l1

    def _compute_gradients(self, X, y, y_pred):
        m = len(X)

        dW = (
            (2 / m) * X.T @ (y_pred - y)
            + self.alpha * np.sign(self.W)
        )

        db = (2 / m) * np.sum(y_pred - y)

        return dW, db

    def fit(self, X, y):

        self.W = np.zeros((X.shape[1], 1))
        self.b = np.zeros((1, 1))

        self.losses = []

        for _ in range(self.epochs):

            y_pred = X @ self.W + self.b

            dW, db = self._compute_gradients(
                X,
                y,
                y_pred
            )

            self.W -= self.lr * dW
            self.b -= self.lr * db

            self.losses.append(
                self._compute_loss(y, y_pred)
            )

        return self

    def predict(self, X):
        return X @ self.W + self.b

    def r2_score(self, y, y_pred):

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)