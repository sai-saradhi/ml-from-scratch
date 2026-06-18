import numpy as np

class PCA:

    def __init__(self, n_components = 2, random_state=42):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        self.mean_ = np.mean(X, axis = 0)
        
        # mean centering
        X_centered = X - self.mean_
        
        # calculating covariance matrix and finding eigenvectors and their values
        C = 1 / (len(X) - 1) * (X_centered.T @ X_centered) 
        self.eigenvalues_, eigenvectors = np.linalg.eigh(C)
        
        # sorting the eigenvectors and eigenvalues in descending order
        indices = np.argsort(-self.eigenvalues_)
        self.eigenvalues_ = self.eigenvalues_[indices]
        eigenvectors = eigenvectors[:,indices]
        self.components_ = eigenvectors[:,:self.n_components]
        self.explained_variance_ratio_ = self.eigenvalues_ / np.sum(self.eigenvalues_)
        
        return None

    def transform(self, X):
        X_centered = X - self.mean_
        
        X_reduced = X_centered @ self.components_
        
        return X_reduced
        

    def fit_transform(self, X):
        self.fit(X)
        X_reduced = self.transform(X)
        
        return X_reduced