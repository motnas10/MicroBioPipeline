import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.colors import Normalize

from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from sklearn.manifold import TSNE
from collections import namedtuple
from emperor import Emperor
from typing import Callable, Tuple, Optional
import torch
from scipy.stats import t
from typing import Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# --------------------------------------------------------------------------------------------------------------
# Correlation and p-value matrices CPU version
def _pair_corr(args):
    """Helper for parallel correlation computation."""
    x, y, corr_func = args
    return corr_func(x, y)

def corr_pval_matrices(
    df: pd.DataFrame,
    axis: int = 0,
    method: str = "pearson",
    corr_func: Optional[Callable[[np.ndarray, np.ndarray], Tuple[float, float]]] = None,
    n_jobs: Optional[int] = 5,
    show_progress:  bool = True,
    ):
    """Compute correlation and p-value matrices in parallel, using matrix symmetry for efficiency."""
    from scipy.stats import pearsonr, spearmanr
    from tqdm import tqdm
    
    if corr_func is None:
        if method == "pearson":
            corr_func = pearsonr
        elif method == "spearman":
            corr_func = spearmanr
        else:
            raise ValueError("Unknown method; provide corr_func or use 'pearson'/'spearman'.")

    if axis == 0:
        labels = df.index
        arrs = [df.loc[idx].values for idx in labels]
    elif axis == 1:
        labels = df.columns
        arrs = [df[col].values for col in labels]
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    n = len(arrs)
    # Prepare only upper triangle tasks
    tasks = [ (arrs[i], arrs[j], corr_func) for i in range(n) for j in range(i, n) ]

    corr_mat = np.zeros((n, n))
    pval_mat = np. zeros((n, n))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Use tqdm to wrap the executor. map results
        results = list(tqdm(
            executor.map(_pair_corr, tasks),
            total=len(tasks),
            desc="Computing correlations",
            disable=not show_progress
        ))

    k = 0
    for i in range(n):
        for j in range(i, n):
            corr, pval = results[k]
            corr_mat[i, j] = corr
            pval_mat[i, j] = pval
            if i != j:  # Fill symmetric position
                corr_mat[j, i] = corr
                pval_mat[j, i] = pval
            k += 1

    corr_df = pd.DataFrame(corr_mat, index=labels, columns=labels)
    pval_df = pd.DataFrame(pval_mat, index=labels, columns=labels)
    return corr_df, pval_df

# --------------------------------------------------------------------------------------------------------------
# Correlation and p-value matrices GPU version
from .correlations import rankdata_torch, pearson_corr_pval, spearman_corr_pval

def corr_pval_matrices_GPU(
    df: pd.DataFrame,
    axis: int = 0,
    method:  str = 'pearson',
    device: str = 'cuda',
    corr_func=None,
    show_progress: bool = True
    ):
    """
    Compute correlation and p-value matrices for a pandas DataFrame using one axis on GPU. 
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    axis : int, default 0
        0: compute pairwise correlations between rows
        1: compute pairwise correlations between columns
    method : str, default "pearson"
        Correlation type. "pearson", "spearman", or provide corr_func.
    device :  str, default 'cuda'
        Device to use: 'cuda' or 'cpu'.
    corr_func : callable, optional
        Custom function: corr_func(x, y) -> (corr, pval).
    show_progress : bool, default True
        Whether to display a progress bar.
    
    Returns
    -------
    tuple: (correlation matrix, p-value matrix)
        corr_df : pd.DataFrame
            Matrix of correlation coefficients.
        pval_df : pd.DataFrame
            Matrix of p-values. 
    """
    if corr_func is None:
        if method == 'pearson':
            corr_func = pearson_corr_pval
        elif method == 'spearman': 
            corr_func = spearman_corr_pval
        else:
            raise ValueError("Unknown method; provide a corr_func or use 'pearson'/'spearman'.")

    if axis == 0:
        labels = df.index
        arrs = [df.loc[idx].values for idx in labels]
    elif axis == 1:
        labels = df.columns
        arrs = [df[col].values for col in labels]
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    arrs_torch = [torch.tensor(a, device=device, dtype=torch.float32) for a in arrs]
    n = len(arrs_torch)
    corr_mat = torch.zeros((n, n), device=device)
    pval_mat = torch.zeros((n, n), device=device)

    # Calculate total number of unique pairs (including diagonal)
    total_pairs = (n * (n + 1)) // 2
    
    # Create progress bar
    pbar = tqdm(total=total_pairs, desc="Computing correlations", disable=not show_progress)
    
    for i in range(n):
        for j in range(i, n):
            x = arrs_torch[i]
            y = arrs_torch[j]
            corr, pval = corr_func(x, y)
            corr_mat[i, j] = corr
            pval_mat[i, j] = pval
            if i != j:
                corr_mat[j, i] = corr
                pval_mat[j, i] = pval
            
            pbar.update(1)
    
    pbar.close()

    corr_np = corr_mat.cpu().numpy()
    pval_np = pval_mat.cpu().numpy()
    corr_df = pd.DataFrame(corr_np, index=labels, columns=labels)
    pval_df = pd.DataFrame(pval_np, index=labels, columns=labels)
    return corr_df, pval_df

# --------------------------------------------------------------------------------------------------------------
# Null correlation distribution with progress bar
def null_corr_distribution(
    df: pd.DataFrame,
    m: int,
    axis: int = 0,
    use_gpu: bool = False,
    method: str = "pearson",
    corr_func: Optional[Callable[[np.ndarray, np.ndarray], Tuple[float, float]]] = None
    ):
    """
    Generate m null models by random shuffling of each row or column of a DataFrame independently,
    compute correlation matrices for each null model, and return mean and std deviation matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    m : int
        Number of null models to generate.
    axis : int, default 0
        If 0, shuffle each row independently; if 1, shuffle each column independently.
    use_gpu : bool, default False
        If True, use corr_pval_matrices_GPU. Otherwise use corr_pval_matrices.

    Returns
    -------
    mean_corr : np.ndarray
        Mean of correlation matrices over m null models.
    sd_corr : np.ndarray
        Standard deviation of correlation matrices over m null models.
    """

    # Store correlation matrices here
    corr_matrices = []

    if use_gpu:
        corr_function = corr_pval_matrices_GPU
    else:
        corr_function = corr_pval_matrices

    # Add tqdm to the for loop for progress visualization
    for _ in tqdm(range(m), desc="Generating null models"):
        df_shuf = df.copy()
        if axis == 0:
            # Shuffle each row independently
            shuf_array = np.apply_along_axis(np.random.permutation, axis=1, arr=df.values)
            df_shuf = pd.DataFrame(shuf_array, index=df.index, columns=df.columns)
        else:
            # Shuffle each column independently
            shuf_array = np.apply_along_axis(np.random.permutation, axis=0, arr=df.values)
            df_shuf = pd.DataFrame(shuf_array, index=df.index, columns=df.columns)

        # Calculate correlation; function should return a matrix (numpy or DataFrame)
        if corr_func is None:
            corr_matrix = corr_function(df_shuf, axis=axis, method=method)[0]
        else:
            corr_matrix = corr_func(df_shuf, axis=axis, method=method)[0]

        # If DataFrame, convert to np.ndarray
        if isinstance(corr_matrix, pd.DataFrame):
            corr_matrix = corr_matrix.values
        corr_matrices.append(corr_matrix)

    # Stack and compute mean and sd
    corr_tensor = np.stack(corr_matrices, axis=0)
    mean_corr = np.mean(corr_tensor, axis=0)
    sd_corr = np.std(corr_tensor, axis=0)

    # Convert back to DataFrame, set index and head based on axis
    if axis == 0:
        mean_corr = pd.DataFrame(mean_corr, index=df.index, columns=df.index)
        sd_corr = pd.DataFrame(sd_corr, index=df.index, columns=df.index)
    else:
        mean_corr = pd.DataFrame(mean_corr, index=df.columns, columns=df.columns)
        sd_corr = pd.DataFrame(sd_corr, index=df.columns, columns=df.columns)

    return mean_corr, sd_corr

# --------------------------------------------------------------------------------------------------------------
# Matrix significance assessment
def matrix_significance_df(corr_data, mean_null, sd_null, device='cpu', p_thresh=1):
    """
    Assess the significance of each entry in corr_data vs null using pandas DataFrames.

    Parameters:
        corr_data : pd.DataFrame, shape (N, N)
        mean_null : pd.DataFrame, shape (N, N)
        sd_null   : pd.DataFrame, shape (N, N)
        device    : 'cpu' or 'cuda'
        p_thresh  : threshold for significance

    Returns:
        z_scores    : pd.DataFrame, shape (N, N)
        p_values    : pd.DataFrame, shape (N, N)
        significant : pd.DataFrame (dtype=bool), shape (N, N)
    """
    # Store index/columns
    idx = corr_data.index
    cols = corr_data.columns

    # Convert to numpy arrays (float32)
    arr_corr = corr_data.values.astype(np.float32)
    arr_mean = mean_null.values.astype(np.float32)
    arr_sd   = sd_null.values.astype(np.float32)

    # Move to torch tensors
    t_corr = torch.as_tensor(arr_corr, device=device)
    t_mean = torch.as_tensor(arr_mean, device=device)
    t_sd   = torch.as_tensor(arr_sd, device=device)

    # Prevent division by zero
    t_sd_safe = torch.where(t_sd == 0, torch.tensor(1e-10, device=device), t_sd)

    # Z-score
    z_scores = (t_corr - t_mean) / t_sd_safe

    # Two-tailed p-value, using normal CDF
    p_values = 2 * (1 - 0.5 * (1 + torch.erf(torch.abs(z_scores) / torch.sqrt(torch.tensor(2.0, device=device)))))

    # Significant mask
    significant = p_values < p_thresh

    # Convert back to numpy/pandas DataFrames
    z_scores_df    = pd.DataFrame(z_scores.cpu().numpy(), index=idx, columns=cols)
    p_values_df    = pd.DataFrame(p_values.cpu().numpy(), index=idx, columns=cols)
    significant_df = pd.DataFrame(significant.cpu().numpy(), index=idx, columns=cols, dtype=bool)

    return z_scores_df, p_values_df, significant_df