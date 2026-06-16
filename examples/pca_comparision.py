import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA as SklearnPCA

from mlfs.decomposition.PCA import PCA


def main():

    # Load dataset
    wine = load_wine()
    X = wine.data

    # Create PCA objects
    my_pca = PCA(n_components=2)
    sklearn_pca = SklearnPCA(n_components=2)

    # Transform data
    X_my = my_pca.fit_transform(X)
    X_sklearn = sklearn_pca.fit_transform(X)

    # Shapes
    print("=" * 50)
    print("SHAPES")
    print("=" * 50)

    print("My PCA Shape      :", X_my.shape)
    print("Sklearn PCA Shape :", X_sklearn.shape)

    # Explained variance ratio
    print("\n" + "=" * 50)
    print("EXPLAINED VARIANCE RATIO")
    print("=" * 50)

    print("My PCA:")
    print(my_pca.explained_variance_ratio_[:2])

    print("\nSklearn PCA:")
    print(sklearn_pca.explained_variance_ratio_)

    # Components
    print("\n" + "=" * 50)
    print("COMPONENTS")
    print("=" * 50)

    print("My PCA Components:")
    print(my_pca.components_)

    print("\nSklearn PCA Components:")
    print(sklearn_pca.components_)

    # Variance ratio comparison
    print("\n" + "=" * 50)
    print("VARIANCE CHECK")
    print("=" * 50)

    variance_match = np.allclose(
        my_pca.explained_variance_ratio_[:2],
        sklearn_pca.explained_variance_ratio_,
        atol=1e-3
    )

    print("Variance Ratio Match:", variance_match)

    # Reduced data comparison
    print("\n" + "=" * 50)
    print("REDUCED DATA")
    print("=" * 50)

    print("My PCA First 5 Rows:")
    print(X_my[:5])

    print("\nSklearn PCA First 5 Rows:")
    print(X_sklearn[:5])

    print(
        "\nNote: Signs may differ because eigenvectors "
        "can be multiplied by -1 and still be valid."
    )


if __name__ == "__main__":
    main()