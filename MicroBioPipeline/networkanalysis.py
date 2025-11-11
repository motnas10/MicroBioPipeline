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
import numpy as np
import pandas as pd
import re
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import pandas as pd
import networkx as nx


# --------------------------------------------------------------------------------------------------------------
# Define generalized statistical comparison function
def node_attributes_extraction(
    df,
    feature_cols,              # List[str]: Columns with numeric features/taxa/data
    metadata_cols,             # List[str]: Columns with metadata/group info, NOT features
    comparisons,               # Dict[str, (mask1, mask2)]: Comparison logic per label
    annotation_extractors,     # Dict[str, (str, int)]: col_name -> (sep, pos) for annotation extraction
    stat_func=mannwhitneyu,    # Function to run stats: stat_func(x, y, **kwargs)
    p_adj_method='fdr_bh',     # Multipletest correction method
    stat_kwargs=None,          # Extra kwargs for stat_func
    pval_symbol_func=None,     # Optional: Function to convert p-value to markers/significance
):
    """
    Perform generalized group-wise statistical comparisons for numeric feature columns in a dataframe,
    extract feature annotations via split/index rules, and summarize results including stats and multiple-testing corrections.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing both data columns and metadata columns.
    feature_cols : List[str]
        List of column names in `df` representing numeric features/taxa/data for statistical analysis.
    metadata_cols : List[str]
        List of column names representing metadata/grouping columns (not included in the main analysis).
    comparisons : Dict[str, Tuple[pd.Series, pd.Series]]
        Dictionary mapping comparison labels to a tuple of boolean masks (mask1, mask2), defining two groups to compare.
        Each mask should be same length as df and indicate which rows go in group1 and group2.
    annotation_extractors : Dict[str, Tuple[str, int]]
        Dict mapping output annotation column names to rules for annotation extraction from feature names.
        The rule is (sep, pos): split the feature string by `sep`, take entry at position `pos` (0-based).
        Example: {'phylum': ('.', 0)} or {'region': ('_', 1)}, etc.
    stat_func : callable, default: scipy.stats.mannwhitneyu
        Function to compute per-feature statistics between two groups, e.g., scipy.stats.mannwhitneyu, ttest_ind, etc.
    p_adj_method : str, default: 'fdr_bh'
        Method passed to statsmodels.stats.multitest.multipletests for multiple testing correction.
    stat_kwargs : dict, optional
        Extra keyword arguments passed to stat_func.
    pval_symbol_func : callable, optional
        Function converting a p-value to a significance marker/string, e.g. "***" for p < 0.001.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame indexed by feature, with columns:
            - 1+ annotation columns (per annotation_extractors)
            - For each comparison:
                - stat_<label>: test statistic
                - p_value_<label>: p-value
                - signif_<label>: significance marker (if pval_symbol_func is set)
                - padj_<label>: multiple testing corrected p-value
                - signif_adj_<label>: significance marker for adjusted p-value
                - mean1_<label>: mean feature value for group1 (comparison's mask1)
                - mean2_<label>: mean feature value for group2 (comparison's mask2)
                - log10_FC_<label>: log10 fold-change (mean2 / mean1, for this comparison)
            - mean_global: mean feature value over all rows
    """

    stat_kwargs = stat_kwargs or dict(alternative='two-sided')
    results = {}
    for label, (mask1, mask2) in comparisons.items():
        stat_values = np.zeros(len(feature_cols))
        p_values = np.zeros(len(feature_cols))
        for i, feat in enumerate(feature_cols):
            data1 = df.loc[mask1, feat].dropna()
            data2 = df.loc[mask2, feat].dropna()
            if len(data1) == 0 or len(data2) == 0:
                stat_values[i] = np.nan
                p_values[i] = np.nan
                continue
            try:
                stat, pval = stat_func(data1, data2, **stat_kwargs)
            except Exception:
                stat, pval = np.nan, np.nan
            stat_values[i] = stat
            p_values[i] = pval
        # Multiple testing correction
        valid = ~np.isnan(p_values)
        corrected_p = np.full_like(p_values, np.nan, dtype=float)
        if np.sum(valid) > 0:
            _, pvals_corrected, _, _ = multipletests(p_values[valid], method=p_adj_method)
            corrected_p[valid] = pvals_corrected
        results[label] = {
            'stat': stat_values,
            'p_value': p_values,
            'padj': corrected_p
        }

    # Annotation extraction
    annotation_df = {}
    for col_name, (sep, pos) in annotation_extractors.items():
        vals = []
        for val in feature_cols:
            parts = val.split(sep)
            try:
                vals.append(parts[pos])
            except IndexError:
                vals.append("unknown")
        annotation_df[col_name] = vals

    results_df = pd.DataFrame(annotation_df)
    results_df.insert(0, 'feature', feature_cols)

    # Add results and stats for each comparison
    for label, res in results.items():
        results_df[f'stat_{label}'] = res['stat']
        results_df[f'p_value_{label}'] = res['p_value']
        if pval_symbol_func:
            results_df[f'signif_{label}'] = results_df[f'p_value_{label}'].apply(pval_symbol_func)
        results_df[f'padj_{label}'] = res['padj']
        if pval_symbol_func:
            results_df[f'signif_adj_{label}'] = results_df[f'padj_{label}'].apply(pval_symbol_func)

        # Compute means for each group
        mask1, mask2 = comparisons[label]
        mean1 = df.loc[mask1, feature_cols].mean(axis=0).to_numpy(dtype=float)
        mean2 = df.loc[mask2, feature_cols].mean(axis=0).to_numpy(dtype=float)
        results_df[f'mean_{label.split("_vs_")[0]}'] = mean1
        results_df[f'mean_{label.split("_vs_")[1]}'] = mean2

        # Per-comparison logFC
        logFC = np.log10(mean2 / mean1)
        logFC = np.where(np.isinf(logFC), np.sign(logFC)*5, logFC)  # Replace +/-inf with +/-5
        logFC = np.nan_to_num(logFC, nan=0)
        l = label.split("_vs_")[1] + "_vs_" + label.split("_vs_")[0]
        results_df[f'log10_FC_{l}'] = logFC

    # Global mean for reference
    results_df['mean_global'] = df[feature_cols].mean(axis=0).to_numpy(dtype=float)
    results_df.set_index('feature', inplace=True)
    return results_df

# --------------------------------------------------------------------------------------------------------------
# Build a signed weighted network from correlation and p-value matrices
def build_signed_weighted_network(corr_mat, p_mat, thr, node_attr=None,
                                 edgefile='edge_list.csv', nodefile='node_attr.csv'):
    """
    Build a weighted, signed network from a correlation matrix, keeping only those with p-value below thr.

    Parameters:
    corr_mat (pd.DataFrame): Correlation matrix
    p_mat (pd.DataFrame): p-value/significance matrix (same shape as corr_mat)
    thr (float): significance threshold; only include edges with p < thr
    node_attr (pd.DataFrame or dict, optional): node attributes. Index/keys should be node names
    edgefile (str): path to save edge list .csv
    nodefile (str): path to save node attributes .csv
    
    Returns:
    nx.Graph: Weighted, signed, undirected graph
    """
    # Mask correlations by p-value threshold
    mask = p_mat < thr
    filtered_corr = corr_mat.where(mask, 0)  # set values to 0 if p >= thr

    # Build networkx graph
    G = nx.Graph()
    nodes = list(filtered_corr.index)
    G.add_nodes_from(nodes)

    # Add edges
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if j <= i: continue  # avoid duplicates + self-loops
            weight = filtered_corr.iloc[i, j]
            if weight != 0:
                G.add_edge(node_i, node_j, weight=weight, sign='+' if weight > 0 else '-')

    # Add node attributes if provided
    if node_attr is not None:
        if isinstance(node_attr, pd.DataFrame):
            attr_dict = node_attr.to_dict(orient='index')
        elif isinstance(node_attr, dict):
            attr_dict = node_attr
        else:
            raise ValueError("node_attr must be DataFrame or dict")
        nx.set_node_attributes(G, attr_dict)

        # Save node attribute table
        node_df = pd.DataFrame.from_dict(attr_dict, orient='index')
        node_df.to_csv(nodefile)

    else:
        # Save only node names
        pd.DataFrame(nodes, columns=['node']).to_csv(nodefile, index=False)

    # Save edge list: node1, node2, weight, sign
    edge_data = [{'source': u, 'target': v, **d} for u, v, d in G.edges(data=True)]
    edge_df = pd.DataFrame(edge_data)
    edge_df.to_csv(edgefile, index=False)

    return G