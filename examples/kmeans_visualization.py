import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans

from mlfs.cluster.Kmeans import KMeans

# Generate data
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Train models
my_kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

sklearn_kmeans = SklearnKMeans(
    n_clusters=3,
    init="random",
    n_init=1,
    random_state=42
)

my_labels = my_kmeans.fit_predict(X)
sklearn_labels = sklearn_kmeans.fit_predict(X)

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# My KMeans
axes[0].scatter(
    X[:, 0],
    X[:, 1],
    c=my_labels
)

axes[0].scatter(
    my_kmeans.centroids_[:, 0],
    my_kmeans.centroids_[:, 1],
    marker="X",
    s=300
)

axes[0].set_title("My KMeans")

# Sklearn KMeans
axes[1].scatter(
    X[:, 0],
    X[:, 1],
    c=sklearn_labels
)

axes[1].scatter(
    sklearn_kmeans.cluster_centers_[:, 0],
    sklearn_kmeans.cluster_centers_[:, 1],
    marker="X",
    s=300
)

axes[1].set_title("Sklearn KMeans")

plt.tight_layout()
plt.show()