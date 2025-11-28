import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def compute_dim_reduction(
    df,
    method='pcoa',
    n_components=2,
    random_state=None,
    use_distance_matrix=True,
    metric='euclidean',
    standardize=True,
    **kwargs
):
    """
    General framework for dimensionality reduction: PCA, PCoA, tSNE, UMAP.

    Parameters:
    - df: pandas DataFrame. Rows=samples, columns=features.
    - method: str. One of 'pca', 'pcoa', 'tsne', 'umap'.
    - n_components: dimensionality of output.
    - random_state: (optional) for reproducibility.
    - use_distance_matrix: bool. Relevant for PCoA, tSNE, UMAP.
    - metric: Distance metric as accepted by sklearn/scipy pairwise_distances.
    - standardize: Whether to standardize numeric columns.
    - kwargs: passed as additional parameters to the underlying method.

    Returns:
    - coords: n_samples x n_components
    - explained_variance: list (if available; e.g., PCA, PCoA); None if not.
    - loadings_df: pd.DataFrame of feature loadings [n_features x n_components] if available, else None.
    """

    # Numeric data only
    data = df.select_dtypes(include=np.number)
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
    else:
        X = data.values

    # Distance matrix if needed
    if use_distance_matrix:
        dist = pairwise_distances(X, metric=metric)
    else:
        dist = None

    # PCA
    if method.lower() == 'pca':
        if use_distance_matrix:
            raise ValueError("PCA does not use a distance matrix!")
        pca = PCA(n_components=n_components, random_state=random_state, **kwargs)
        coords = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_.tolist()
        # Feature loadings: correl. between each feature and each component
        loadings = np.empty((X.shape[1], n_components))
        for i in range(n_components):
            for j in range(X.shape[1]):
                loadings[j, i] = np.corrcoef(X[:, j], coords[:, i])[0, 1]
        loadings_df = pd.DataFrame(loadings, index=data.columns, columns=[f'PCA{i+1}' for i in range(n_components)])
        return coords, explained, loadings_df

    # PCoA
    elif method.lower() == 'pcoa':
        if dist is None:
            # fallback to euclidean
            dist = pairwise_distances(X, metric="euclidean")
        n = dist.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (dist ** 2) @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        coords = eigvecs[:, :n_components] * np.sqrt(np.abs(eigvals[:n_components]))
        positive_eigvals = eigvals[eigvals > 0]
        total_variance = np.sum(positive_eigvals)
        explained = [(eigvals[i] / total_variance) if eigvals[i] > 0 else 0 for i in range(n_components)]
        # Feature loadings
        loadings = np.empty((X.shape[1], n_components))
        for i in range(n_components):
            for j in range(X.shape[1]):
                loadings[j, i] = np.corrcoef(X[:, j], coords[:, i])[0, 1]
        loadings_df = pd.DataFrame(loadings, index=data.columns, columns=[f'PCoA{i+1}' for i in range(n_components)])
        return coords.real, explained, loadings_df

    # tSNE
    elif method.lower() == 'tsne':
        if use_distance_matrix:
            # pairwise distances, then fit TSNE with precomputed
            tsne = TSNE(n_components=n_components, random_state=random_state, metric="precomputed", init="random", **kwargs)
            coords = tsne.fit_transform(dist)
        else:
            tsne = TSNE(n_components=n_components, random_state=random_state, metric=metric, **kwargs)
            coords = tsne.fit_transform(X)
        return coords, [None]*n_components, None

    # UMAP
    # UMAP
    elif method.lower() == 'umap':
        umap_kwargs = dict(n_components=n_components, random_state=random_state, metric=metric)
        umap_kwargs.update(kwargs)
        if use_distance_matrix:
            reducer = umap.UMAP(**umap_kwargs)
            coords = reducer.fit_transform(dist)
        else:
            reducer = umap.UMAP(**umap_kwargs)
            coords = reducer.fit_transform(X)
        # UMAP does not provide explained variance/loadings
        return coords, [None]*n_components, None

    else:
        raise ValueError("method must be one of: 'pca', 'pcoa', 'tsne', 'umap'")
