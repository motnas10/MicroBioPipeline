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