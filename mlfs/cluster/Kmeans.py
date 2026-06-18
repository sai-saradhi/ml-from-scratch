import numpy as np

class KMeans:

    def __init__(self, n_clusters = 3, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol= tol
        self.centroids_ = None
        self.labels_ = None
        self.random_state = random_state
    
    def fit(self, X):
        np.random.seed(self.random_state)

        # Initialize centroids
        self.centroids_ = X[
            np.random.choice(
                X.shape[0],
                size=self.n_clusters,
                replace=False
            )
        ]

        for _ in range(self.max_iters):

            # Save old centroids for convergence check
            old_centroids = self.centroids_.copy()

            # Assignment Step
            diff = self.centroids_[:, np.newaxis, :] - X[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)

            self.labels_ = np.argmin(
                dist_matrix.T,
                axis=1
            )

            # Update Step
            new_centroids = []

            for c in range(self.n_clusters):

                mask = (self.labels_ == c)

                # Handle empty cluster
                if np.sum(mask) == 0:
                    new_centroids.append(
                        self.centroids_[c]
                    )
                    continue

                cluster_points = X[mask]

                centroid = np.mean(
                    cluster_points,
                    axis=0
                )

                new_centroids.append(centroid)

            new_centroids = np.array(new_centroids)

            # Convergence Check
            if np.all(
                np.abs(new_centroids - old_centroids)
                < self.tol
            ):
                self.centroids_ = new_centroids
                break

            self.centroids_ = new_centroids

        return self
    
    def predict(self, X):
        diff = self.centroids_[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2) 
        self.labels_ = np.argmin(dist_matrix.T, axis=1)
        
        return self.labels_

    def fit_predict(self, X):
        self.fit(X)
        self.predict(X)
        
        return self.labels_
        

