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
import pandas as pd
from scipy.stats import t

# --------------------------------------------------------------------------------------------------------------
# Correlation functions using PyTorch
def rankdata_torch(x: torch.Tensor):
    """
    Rank the data in a 1D torch tensor, handling ties by assigning average ranks.
    Args:
        x (torch.Tensor): 1D tensor of data to rank.
    Returns:
        torch.Tensor: 1D tensor of ranks.
    """
    sorted_x, indices = torch.sort(x)
    ranks = torch.zeros_like(x, dtype=torch.float32)
    unique, inverse, counts = torch.unique(sorted_x, return_inverse=True, return_counts=True)
    cum_counts = torch.cumsum(counts, 0) - counts
    for val, start, count in zip(unique, cum_counts, counts):
        idxs = (sorted_x == val).nonzero(as_tuple=True)[0]
        avg_rank = torch.mean(torch.arange(start+1, start+count+1, device=x.device).float())
        ranks[indices[idxs]] = avg_rank
    return ranks

def pearson_corr_pval(x: torch.Tensor, y: torch.Tensor):
    """
    Compute Pearson correlation coefficient and p-value between two 1D torch tensors.
    Args:
        x (torch.Tensor): 1D tensor.
        y (torch.Tensor): 1D tensor.
    Returns:
        tuple: (correlation coefficient, p-value)
    """
    xm = x - x.mean()
    ym = y - y.mean()
    r = (xm * ym).sum() / torch.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    n = x.size(0)
    r = torch.clamp(r, min=-0.999999, max=0.999999)
    t_stat = r * torch.sqrt((n - 2) / (1 - r**2))
    # Use scipy for p-value; move t_stat to CPU & numpy
    t_stat_np = float(torch.abs(t_stat).cpu().numpy())
    pval = 2 * (1 - t.cdf(t_stat_np, df=n-2))
    return r.item(), float(pval)

def spearman_corr_pval(x: torch.Tensor, y: torch.Tensor):
    """
    Compute Spearman correlation coefficient and p-value between two 1D torch tensors.
    Args:
        x (torch.Tensor): 1D tensor.
        y (torch.Tensor): 1D tensor.
    Returns:
        tuple: (correlation coefficient, p-value)
    """
    rx = rankdata_torch(x)
    ry = rankdata_torch(y)
    return pearson_corr_pval(rx, ry)

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def pvalues_correction(pvalue_df, method='fdr_bh', alpha=0.05):
    """
    Apply multiple testing correction to p-values in a pandas DataFrame.
    
    Parameters
    ----------
    pvalue_df : pd.DataFrame
        DataFrame containing p-values. Can be any shape.
    method : str, default='fdr_bh'
        Correction method to apply. Options:
        - 'bonferroni', 'sidak', 'holm'
        - 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'
        - 'none': no correction (returns original p-values)
    alpha : float, default=0.05
        Threshold for significance.
    
    Returns
    -------
    dict
        - 'corrected_pvalues': DataFrame of corrected p-values
        - 'rejected': DataFrame of boolean mask where True = significant
        - 'method': correction method
        - 'alpha': threshold
        - 'n_tests': number of valid tests
    """
    # Flatten and track NaNs
    pvals_flat = pvalue_df.values.flatten()
    valid_mask = ~np.isnan(pvals_flat)
    valid_pvals = pvals_flat[valid_mask]
    n_tests = len(valid_pvals)
    
    if n_tests == 0:
        raise ValueError("No valid p-values found in the DataFrame")
    
    # Initialize arrays
    corrected_flat = pvals_flat.copy()
    rejected_flat = np.zeros_like(pvals_flat, dtype=bool)
    
    # Apply correction
    if method == 'none':
        corrected_valid = valid_pvals
        rejected_valid = valid_pvals <= alpha
    
    elif method == 'bonferroni':
        corrected_valid = np.minimum(valid_pvals * n_tests, 1.0)
        rejected_valid = corrected_valid <= alpha
    
    elif method == 'sidak':
        corrected_valid = 1 - (1 - valid_pvals) ** n_tests
        rejected_valid = corrected_valid <= alpha
    
    elif method == 'holm':
        sort_idx = np.argsort(valid_pvals)
        sorted_pvals = valid_pvals[sort_idx]
        corrected_sorted = np.minimum.accumulate(sorted_pvals * np.arange(n_tests, 0, -1))
        corrected_sorted = np.minimum(corrected_sorted, 1.0)
        corrected_valid = np.empty(n_tests)
        corrected_valid[sort_idx] = corrected_sorted
        rejected_valid = corrected_valid <= alpha
    
    elif method in ['fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky']:
        rejected_valid, corrected_valid, _, _ = multipletests(
            valid_pvals, alpha=alpha, method=method.replace('fdr_', '')
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fill arrays
    corrected_flat[valid_mask] = corrected_valid
    rejected_flat[valid_mask] = rejected_valid
    
    # Reshape to original
    corrected_df = pd.DataFrame(corrected_flat.reshape(pvalue_df.shape),
                                index=pvalue_df.index,
                                columns=pvalue_df.columns)
    rejected_df = pd.DataFrame(rejected_flat.reshape(pvalue_df.shape),
                               index=pvalue_df.index,
                               columns=pvalue_df.columns)
    
    return {
        'corrected_pvalues': corrected_df,
        'rejected': rejected_df,
        'method': method,
        'alpha': alpha,
        'n_tests': n_tests
    }


# --------------------------------------------------------------------------------------------------------------
# Correlation computation
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, spearmanr, chi2_contingency
from scipy.spatial.distance import braycurtis, euclidean, cityblock, cosine, correlation, hamming
from sklearn.metrics import mutual_info_score

def sparcc_correlation(data, eps=1e-6):
    data = np.asarray(data, dtype=float)
    data = np.where(data <= 0, eps, data)
    log_data = np.log(data)
    gm = log_data.mean(axis=1, keepdims=True)
    clr = log_data - gm
    cov = np.cov(clr.T)
    var = np.diag(cov)
    sd = np.sqrt(var)
    sd[sd == 0] = np.nan
    corr = cov / np.outer(sd, sd)
    corr = np.clip(corr, -1, 1)
    return corr

def jaccard_manual(a, b):
    inter = np.sum((a == 1) & (b == 1))
    union = np.sum((a == 1) | (b == 1))
    return inter / union if union > 0 else np.nan

def fisher(a, b):
    table = np.array([
        [np.sum((a == 1) & (b == 1)), np.sum((a == 1) & (b == 0))],
        [np.sum((a == 0) & (b == 1)), np.sum((a == 0) & (b == 0))]
    ])
    _, p = fisher_exact(table, alternative="two-sided")
    return p

def phi(a, b):
    a1b1 = np.sum((a == 1) & (b == 1))
    a1b0 = np.sum((a == 1) & (b == 0))
    a0b1 = np.sum((a == 0) & (b == 1))
    a0b0 = np.sum((a == 0) & (b == 0))
    num = (a1b1 * a0b0) - (a1b0 * a0b1)
    den = np.sqrt((a1b1 + a1b0) * (a1b1 + a0b1) * (a0b1 + a0b0) * (a1b0 + a0b0))
    phi_val = num / den if den != 0 else np.nan
    table = np.array([[a1b1, a1b0], [a0b1, a0b0]])
    chi2, p, _, _ = chi2_contingency(table, correction=False)
    return phi_val, p

def pairwise_correlation(df, metric="jaccard", axis=1, permutations=None, random_state=None):
    """
    Extended version: 
    - 'permutations': int or None. If set, compute permutation-based p-values.
    - The randomization will be done on the *second* series (B).
    """
    similarity_metrics = ["jaccard", "fisher", "phi", "pearson", "spearman", "mutual_info", "sparcc"]
    distance_metrics = ["braycurtis", "euclidean", "cityblock", "cosine", "correlation", "hamming"]

    if metric not in similarity_metrics + distance_metrics:
        raise ValueError(f"Unknown metric: {metric}")

    # orientation
    if axis == 0:
        df = df.T

    if metric in ["jaccard", "fisher", "phi", "hamming", "mutual_info"]:
        data = (df != 0).astype(int)
    else:
        data = df.copy()

    features = data.columns
    n = len(features)
    val = pd.DataFrame(np.zeros((n, n)), index=features, columns=features)
    pval = pd.DataFrame(np.full((n, n), np.nan), index=features, columns=features)

    # Use a RandomState for reproducibility if provided
    rng = np.random.RandomState(random_state)

    # Make metric function
    if metric == "pearson":
        _func = lambda a, b: pearsonr(a, b)[0]
        _pfunc = lambda a, b: pearsonr(a, b)[1]
        mode = "higher"
    elif metric == "spearman":
        _func = lambda a, b: spearmanr(a, b)[0]
        _pfunc = lambda a, b: spearmanr(a, b)[1]
        mode = "higher"
    elif metric == "jaccard":
        _func = jaccard_manual
        mode = "higher"
    elif metric == "fisher":
        _func = lambda a, b: np.nan
        _pfunc = fisher
        mode = "neither"  # Only p-value
    elif metric == "phi":
        _func = lambda a, b: phi(a, b)[0]
        _pfunc = lambda a, b: phi(a, b)[1]
        mode = "both"
    elif metric == "mutual_info":
        _func = lambda a, b: mutual_info_score(a, b) if (np.unique(a).size > 1 and np.unique(b).size > 1) else np.nan
        mode = "higher"
    elif metric == "sparcc":
        data_sparcc = df.values.astype(float)
        corr = sparcc_correlation(data_sparcc)
        val.loc[:, :] = corr
        pval.loc[:, :] = np.nan
        return val, pval
    else:
        dist_funcs = {
            'braycurtis': braycurtis,
            'euclidean': euclidean,
            'cityblock': cityblock,
            'cosine': cosine,
            'correlation': correlation,
            'hamming': hamming
        }
        _func = dist_funcs[metric]
        mode = "lower"

    # Run main loop
    for i in range(n):
        for j in range(i, n):
            a = data.iloc[:, i].values
            b = data.iloc[:, j].values
            # Observed
            if metric == "fisher":
                v = np.nan
                p = _pfunc(a, b)
            elif metric == "phi":
                v, p = phi(a, b)
            elif metric in ["pearson", "spearman"]:
                v = _func(a, b)
                p = _pfunc(a, b)
            else:
                v = _func(a, b)
                p = np.nan

            # Permutation-based p-value
            if permutations is not None and metric not in ["fisher", "phi"]:  # For 'fisher' and 'phi', p-value handled analytically
                perm_vals = []
                for _ in range(permutations):
                    if rng is not None:
                        b_perm = rng.permutation(b)
                    else:
                        b_perm = np.random.permutation(b)
                    try:
                        perm_val = _func(a, b_perm)
                    except Exception:
                        perm_val = np.nan
                    perm_vals.append(perm_val)
                perm_vals = np.asarray(perm_vals)
                if mode == "higher":
                    p_perm = np.mean(perm_vals >= v)
                elif mode == "lower":
                    p_perm = np.mean(perm_vals <= v)
                else:
                    p_perm = np.nan
                p = p_perm

            val.iat[i, j] = val.iat[j, i] = v
            pval.iat[i, j] = pval.iat[j, i] = p

    return val, pval