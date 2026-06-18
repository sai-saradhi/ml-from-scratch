import  numpy as np

class KNN:
    
    def __init__(self, K=3):
        self.K = 3
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        return self
    
    def predict(self, X):
        # calculate euclidean distance to the training points 
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        sorted_dist = np.argsort(dist_matrix)
        
        K_neighbors = self.y_train[sorted_dist[:,:self.K]]
        predictions = []

        for neighbors in K_neighbors:

            vals, counts = np.unique(
                neighbors,
                return_counts=True
            )

            predictions.append(
                vals[np.argmax(counts)]
            )

        return np.array(predictions)
    
