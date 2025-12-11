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
