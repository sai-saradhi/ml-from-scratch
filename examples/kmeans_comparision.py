import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from mlfs.cluster.Kmeans import KMeans

np.random.seed(42)

def main():

    # Load dataset
    X, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )

    # Create Kmeans objects
    my_kmeans = KMeans(n_clusters=3, random_state=42)
    sklearn_kmeans = SklearnKMeans(
        n_clusters=3,
        init="random",
        n_init=1,
        random_state=42
    )

    # Transform data
    X_my = my_kmeans.fit_predict(X)
    X_sklearn = sklearn_kmeans.fit_predict(X)

    # Shapes
    print("=" * 50)
    print("SHAPES")
    print("=" * 50)

    print("My Kmeans Shape      :", X_my.shape)
    print("Sklearn Kmeans Shape :", X_sklearn.shape)

    # Centroids of Clusters 
    print("\n" + "=" * 50)
    print("Centroids of Clusters")
    print("=" * 50)

    print("My Kmeans:")
    print(my_kmeans.centroids_)

    print("\nSklearn Kmeans:")
    print(sklearn_kmeans.cluster_centers_)

    # Labels Produced
    print("\n" + "=" * 50)
    print("Labels")
    print("=" * 50)
    
    labels_check = np.allclose(
        my_kmeans.labels_,
        sklearn_kmeans.labels_,
        atol=1e-3
    )

    print("My Kmeans Lables:")
    print(my_kmeans.labels_)

    print("\nSklearn Kmeans Lables:")
    print(sklearn_kmeans.labels_)
    
    print("labels Match:", labels_check)
    
    score = adjusted_rand_score(
        sklearn_kmeans.labels_,
        my_kmeans.labels_
    )
    print("score : ",score)
    # # Centroid comparison
    # print("\n" + "=" * 50)
    # print("Centroid CHECK")
    # print("=" * 50)

    # centroid_check = np.allclose(
    #     my_kmeans.centroids_,
    #     sklearn_kmeans.cluster_centers_,
    #     atol=1e-3
    # )

    # print("Centroid Match:", centroid_check)


if __name__ == "__main__":
    main()