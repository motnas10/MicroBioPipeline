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

# --------------------------------------------------------------------------------------------------------------
# Updated network plotting module with selectable layouts
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional, Union, Tuple, Dict, List
from itertools import combinations


class NetworkBuilder:
    """Handles network graph construction from correlation data."""
    
    @staticmethod
    def build_network(
        corr_matrix: pd.DataFrame,
        use_sfdp: bool = True,
        layout: Optional[str] = None,
        layout_params: Optional[Dict] = None
    ) -> Tuple[nx.Graph, Dict]:
        """
        Build network graph from correlation matrix.
        
        Assumes the correlation matrix has already been filtered for significance
        and correlation thresholds. Only non-zero/non-NaN correlations will be 
        included as edges.

        Parameters:
            corr_matrix: Pre-filtered correlation matrix
            use_sfdp: Deprecated. Use layout='sfdp' instead.
            layout: Layout algorithm to use. If None, behavior is:
                    - 'sfdp' if use_sfdp is True (default)
                    - 'spring' if use_sfdp is False
                Supported values (case-insensitive): 'sfdp', 'spring', 'kamada_kawai',
                'kamada', 'circular', 'spectral', 'shell', 'random', 'planar'
            layout_params: Optional dict passed to the layout function
                
        Returns:
            Tuple of (Graph, position_dict)
        """
        G = nx.Graph()
        G.add_nodes_from(corr_matrix.columns)
        
        # Add edges from the correlation matrix
        # Only include edges where correlation is non-zero and not NaN
        for col1, col2 in combinations(corr_matrix.columns, 2):
            r = corr_matrix.loc[col1, col2]
            
            # Skip if NaN or zero (assumes pre-filtering)
            if pd.notna(r) and r != 0:
                G.add_edge(col1, col2, weight=abs(r), corr=r)
        
        # Decide layout choice
        if layout is None:
            layout_choice = 'sfdp' if use_sfdp else 'spring'
        else:
            layout_choice = layout.lower()
        
        # Compute layout
        pos = NetworkBuilder._compute_layout(G, layout_choice, layout_params)
        return G, pos
    
    @staticmethod
    def _compute_layout(
        G: nx.Graph, 
        layout: str,
        layout_params: Optional[Dict] = None
    ) -> Dict:
        """Compute node positions using specified layout algorithm.

        layout: name of the layout ('sfdp', 'spring', 'kamada_kawai', 'circular', ...)
        layout_params: optional dict passed to the underlying layout function.
        """
        layout_params = layout_params or {}

        # When graph is empty or single node, return trivial positions
        if G.number_of_nodes() == 0:
            return {}
        if G.number_of_nodes() == 1:
            return {n: (0.0, 0.0) for n in G.nodes()}

        layout = layout.lower()
        
        # Helper to call networkx layout functions while being tolerant of unexpected kwargs
        def _call_layout_fn(fn, G, params):
            if not params:
                return fn(G)
            try:
                return fn(G, **params)
            except TypeError:
                # Some layouts accept different keyword args; try calling without params.
                try:
                    return fn(G)
                except Exception as e:
                    raise

        if layout == 'sfdp':
            try:
                import pygraphviz  # noqa: F401
                A = nx.nx_agraph.to_agraph(G)
                
                # Default SFDP parameters
                default_sfdp = {'overlap': 'scale', 'K': '1.0'}
                sfdp_params = default_sfdp.copy()
                sfdp_params.update(layout_params or {})
                
                A.graph_attr.update(layout='sfdp', **sfdp_params)
                A.layout('sfdp')
                
                pos = {
                    n: tuple(map(float, A.get_node(n).attr['pos'].split(',')))
                    for n in G.nodes()
                }
                return pos
            except Exception as e:
                print(f"⚠️ SFDP layout requested but failed ({e}). Falling back to spring layout.")
                # fall through to spring layout

        # Mapping of friendly layout names to networkx functions
        nx_layouts = {
            'spring': nx.spring_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'kamada': nx.kamada_kawai_layout,
            'circular': nx.circular_layout,
            'spectral': nx.spectral_layout,
            'shell': nx.shell_layout,
            'random': nx.random_layout,
            'planar': nx.planar_layout
        }

        if layout in nx_layouts:
            fn = nx_layouts[layout]
            # Provide some reasonable defaults for spring if not specified
            if fn is nx.spring_layout:
                spring_defaults = {'seed': 42, 'k': None, 'iterations': 50}
                params = spring_defaults.copy()
                params.update(layout_params or {})
                try:
                    return _call_layout_fn(fn, G, params)
                except Exception as e:
                    print(f"⚠️ spring_layout failed ({e}). Using nx.random_layout as last resort.")
                    return nx.random_layout(G)
            else:
                try:
                    return _call_layout_fn(fn, G, layout_params or {})
                except Exception as e:
                    print(f"⚠️ {layout} layout failed ({e}). Falling back to spring layout.")
                    # Try spring as fallback
                    try:
                        spring_defaults = {'seed': 42, 'k': None, 'iterations': 50}
                        params = spring_defaults.copy()
                        params.update(layout_params or {})
                        return _call_layout_fn(nx.spring_layout, G, params)
                    except Exception as e2:
                        print(f"⚠️ spring_layout fallback also failed ({e2}). Using nx.random_layout.")
                        return nx.random_layout(G)
        else:
            # Unknown layout name: warn and fallback to spring
            print(f"⚠️ Unknown layout '{layout}'. Supported layouts: sfdp, spring, kamada_kawai, "
                  "circular, spectral, shell, random, planar. Falling back to spring layout.")
            try:
                spring_defaults = {'seed': 42, 'k': None, 'iterations': 50}
                params = spring_defaults.copy()
                params.update(layout_params or {})
                return _call_layout_fn(nx.spring_layout, G, params)
            except Exception as e:
                print(f"⚠️ spring_layout failed ({e}). Using nx.random_layout.")
                return nx.random_layout(G)


class NodeStyler:
    """Handles node styling (color, size, shape) based on metadata."""
    
    def __init__(
        self,
        nodelist: List,
        metadata: Optional[pd.DataFrame] = None,
        default_color: str = "lightblue",
        default_size: float = 600,
        default_shape: str = "o"
    ):
        self.nodelist = nodelist
        self.metadata = metadata
        self.default_color = default_color
        self.default_size = default_size
        self.default_shape = default_shape
        
    def get_node_colors(
        self, 
        color_by: Optional[str] = None,
        cmap: Union[str, mpl.colors.Colormap] = "viridis"
    ) -> Tuple[Dict, Optional[mpl.cm.ScalarMappable], Optional[List]]:
        """
        Get node colors based on metadata attribute.
        
        Returns:
            Tuple of (color_dict, scalar_mappable_for_colorbar, legend_elements)
        """
        node_colors = {n: self.default_color for n in self.nodelist}
        sm = None
        legend_elements = None
        
        if self.metadata is None or color_by is None or color_by not in self.metadata.columns:
            return node_colors, sm, legend_elements
        
        series = self.metadata[color_by].copy()
        # Ensure index alignment
        series = series.reindex(self.nodelist)
        
        if pd.api.types.is_numeric_dtype(series):
            # Continuous colormap
            valid_vals = series.dropna()
            if len(valid_vals) == 0:
                return node_colors, sm, legend_elements
                
            vmin, vmax = valid_vals.min(), valid_vals.max()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            ccmap = plt.cm.get_cmap(cmap)
            
            node_colors = {
                n: ccmap(norm(series.loc[n])) if pd.notna(series.loc[n]) else ccmap(0.5)
                for n in self.nodelist
            }
            
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=ccmap)
            sm.set_array([])
        else:
            # Categorical colors
            categories = series.dropna().unique()
            if len(categories) == 0:
                return node_colors, sm, legend_elements
                
            palette = plt.cm.get_cmap("tab20", max(2, len(categories)))
            cat_to_color = {cat: palette(i) for i, cat in enumerate(categories)}
            
            node_colors = {
                n: cat_to_color.get(series.loc[n], palette(0)) if pd.notna(series.loc[n]) else palette(0)
                for n in self.nodelist
            }
            
            legend_elements = [
                mpl.lines.Line2D(
                    [0], [0], marker=self.default_shape, color='w',
                    markerfacecolor=cat_to_color[cat],
                    markersize=10, label=str(cat)
                )
                for cat in categories
            ]
        
        return node_colors, sm, legend_elements
    
    def get_node_sizes(
        self, 
        size_by: Optional[str] = None,
        min_size: float = 300,
        max_size: float = 1500
    ) -> Dict:
        """Get node sizes based on metadata attribute."""
        node_sizes = {n: self.default_size for n in self.nodelist}
        
        if self.metadata is None or size_by is None or size_by not in self.metadata.columns:
            return node_sizes
        
        vals = self.metadata[size_by]
        if not pd.api.types.is_numeric_dtype(vals):
            return node_sizes
        
        val_range = vals.max() - vals.min()
        if val_range < 1e-9:
            return node_sizes
        
        vals_scaled = min_size + (max_size - min_size) * (vals - vals.min()) / val_range
        node_sizes = {
            n: vals_scaled.loc[n] if n in vals.index else self.default_size
            for n in self.nodelist
        }
        
        return node_sizes
    
    def get_node_shapes(
        self, 
        marker_by: Optional[str] = None,
        marker_list: Optional[List[str]] = None
    ) -> Tuple[Dict, Optional[List]]:
        """
        Get node shapes based on metadata attribute.
        
        Returns:
            Tuple of (marker_dict, legend_elements)
        """
        node_markers = {n: self.default_shape for n in self.nodelist}
        legend_elements = None
        
        if self.metadata is None or marker_by is None or marker_by not in self.metadata.columns:
            return node_markers, legend_elements
        
        series = self.metadata[marker_by]
        categories = series.dropna().unique()
        
        if marker_list is None:
            marker_list = ['o', 's', '^', 'D', 'v', 'h', '*', 'P', 'X', '<', '>']
        
        cat_to_marker = {
            cat: marker_list[i % len(marker_list)] 
            for i, cat in enumerate(categories)
        }
        
        node_markers = {
            n: cat_to_marker.get(series.loc[n], self.default_shape) if n in series.index else self.default_shape
            for n in self.nodelist
        }
        
        legend_elements = [
            mpl.lines.Line2D(
                [0], [0], marker=cat_to_marker[cat], color='gray',
                markerfacecolor='gray', markersize=10, 
                label=str(cat), linestyle='None'
            )
            for cat in categories
        ]
        
        return node_markers, legend_elements


class NetworkPlotter:
    """Handles network visualization."""
    
    def __init__(
        self,
        G: nx.Graph,
        pos: Dict,
        metric: str = "correlation",
        metadata: Optional[pd.DataFrame] = None
    ):
        self.G = G
        self.pos = pos
        self.metric = metric
        self.metadata = metadata
        self.nodelist = list(G.nodes)
        
    def plot(
        self,
        color_by: Optional[str] = None,
        size_by: Optional[str] = None,
        marker_by: Optional[str] = None,
        default_node_shape: str = "o",
        cmap: Union[str, mpl.colors.Colormap] = "viridis",
        figsize: Tuple[int, int] = (10, 10),
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        edge_width_scale: float = 1.0,
        edge_alpha: float = 0.8,
        node_alpha: float = 0.9,
        node_linewidth: float = 1.0,
        label_font_size: int = 10,
        legend_font_size: int = 10,
        title_font_size: int = 12,
        colorbar_font_size: int = 10,
        show: bool = True,
        default_equal_node_size: bool = True,
        default_node_size: float = 600,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot network with customizable styling.
        
        Args:
            color_by: Metadata column for node colors
            size_by: Metadata column for node sizes
            marker_by: Metadata column for node shapes
            default_node_shape: Default marker shape
            cmap: Colormap for edges and continuous node colors
            figsize: Figure size
            fig, ax: Existing figure/axes (for subplots)
            title: Plot title
            edge_width_scale: Scale factor for edge widths
            edge_alpha: Edge transparency
            node_alpha: Node transparency
            node_linewidth: Node border width
            label_font_size: Font size for node labels
            legend_font_size: Font size for legend text
            title_font_size: Font size for plot title
            colorbar_font_size: Font size for colorbar
            show: Whether to display the plot
            default_equal_node_size: If True and no marker_by categorical grouping is provided,
                                     force all nodes to the same size (styler.default_size).
        
        Returns:
            Tuple of (figure, axes)
        """
        # Create figure if needed
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Initialize styler
        styler = NodeStyler(
            self.nodelist, 
            self.metadata,
            default_size=default_node_size,
            default_shape=default_node_shape
        )
        
        # Get node styling
        node_colors, color_sm, color_legend = styler.get_node_colors(color_by, cmap)
        node_sizes = styler.get_node_sizes(size_by)
        node_markers, marker_legend = styler.get_node_shapes(marker_by)
        
        # If the user wants uniform sizes when no categorical grouping is provided,
        # and marker_by is not provided (or not in metadata), override node_sizes:
        if default_equal_node_size and (marker_by is None or self.metadata is None or marker_by not in self.metadata.columns):
            node_sizes = {n: styler.default_size for n in self.nodelist}
        
        # Draw edges
        self._draw_edges(ax, cmap, edge_width_scale, edge_alpha)
        
        # Draw nodes
        self._draw_nodes(
            ax, node_colors, node_sizes, node_markers,
            marker_by, node_alpha, node_linewidth
        )
        
        # Draw labels
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=label_font_size)
        
        # Add edge colorbar
        edge_sm = self._create_edge_scalar_mappable(cmap)
        cbar = fig.colorbar(edge_sm, ax=ax)
        cbar.set_label(f"{self.metric.capitalize()}", fontsize=colorbar_font_size)
        cbar.ax.tick_params(labelsize=colorbar_font_size)
        
        # Add node colorbar if continuous coloring
        if color_sm is not None:
            cbar_nodes = fig.colorbar(color_sm, ax=ax, pad=0.02)
            cbar_nodes.set_label(color_by if color_by else "Value", fontsize=colorbar_font_size)
            cbar_nodes.ax.tick_params(labelsize=colorbar_font_size)
        
        # Add legends
        self._add_legends(
            ax, color_legend, marker_legend, 
            color_by, marker_by, legend_font_size
        )
        
        ax.axis('off')

        if title is not None:
            fig.suptitle(title, fontsize=title_font_size)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return fig, ax
    
    def _draw_edges(
        self, 
        ax: plt.Axes, 
        cmap: Union[str, mpl.colors.Colormap],
        width_scale: float,
        alpha: float
    ):
        """Draw network edges."""
        edges = list(self.G.edges(data=True))
        edge_corr_values = [d['corr'] for (_, _, d) in edges]
        edge_widths = [abs(d['corr']) * width_scale for (_, _, d) in edges]
        
        edge_cmap = plt.cm.get_cmap(cmap)
        
        nx.draw_networkx_edges(
            self.G, self.pos, ax=ax,
            edgelist=list(self.G.edges),
            edge_color=edge_corr_values,
            edge_cmap=edge_cmap,
            edge_vmin=-1, edge_vmax=1,
            width=edge_widths,
            alpha=alpha
        )
    
    def _draw_nodes(
        self,
        ax: plt.Axes,
        node_colors: Dict,
        node_sizes: Dict,
        node_markers: Dict,
        marker_by: Optional[str],
        alpha: float,
        linewidth: float
    ):
        """Draw network nodes with proper styling."""
        if marker_by is not None and self.metadata is not None and marker_by in self.metadata.columns:
            # Draw by marker category
            series = self.metadata[marker_by]
            for cat in series.dropna().unique():
                nodes_in_cat = [
                    n for n in self.nodelist 
                    if n in series.index and series.loc[n] == cat
                ]
                if not nodes_in_cat:
                    continue
                    
                nx.draw_networkx_nodes(
                    self.G, self.pos, nodelist=nodes_in_cat, ax=ax,
                    node_size=[node_sizes[n] for n in nodes_in_cat],
                    node_color=[node_colors[n] for n in nodes_in_cat],
                    node_shape=node_markers[nodes_in_cat[0]],
                    alpha=alpha,
                    linewidths=linewidth
                )
        else:
            # Draw all nodes with same marker
            nx.draw_networkx_nodes(
                self.G, self.pos, nodelist=self.nodelist, ax=ax,
                node_size=[node_sizes[n] for n in self.nodelist],
                node_color=[node_colors[n] for n in self.nodelist],
                node_shape=node_markers[self.nodelist[0]],
                alpha=alpha,
                linewidths=linewidth
            )
    
    def _create_edge_scalar_mappable(
        self, 
        cmap: Union[str, mpl.colors.Colormap]
    ) -> mpl.cm.ScalarMappable:
        """Create scalar mappable for edge colorbar."""
        edge_norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        edge_cmap = plt.cm.get_cmap(cmap)
        sm = mpl.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
        sm.set_array([])
        return sm
    
    def _add_legends(
        self,
        ax: plt.Axes,
        color_legend: Optional[List],
        marker_legend: Optional[List],
        color_by: Optional[str],
        marker_by: Optional[str],
        font_size: int
    ):
        """Add color and marker legends."""
        # Debug: Check what legends we have
        has_color_legend = color_legend is not None and len(color_legend) > 0
        has_marker_legend = marker_legend is not None and len(marker_legend) > 0
        
        if has_color_legend and has_marker_legend:
            # Combine both legends
            handles = color_legend + marker_legend
            legend = ax.legend(
                handles=handles, 
                loc="best", 
                frameon=True,
                fontsize=font_size
            )
            if legend:
                legend.set_zorder(1000)
        elif has_color_legend:
            legend = ax.legend(
                handles=color_legend, 
                title=color_by, 
                loc="upper left", 
                frameon=True,
                fontsize=font_size,
                title_fontsize=font_size
            )
            if legend:
                legend.set_zorder(1000)
        elif has_marker_legend:
            legend = ax.legend(
                handles=marker_legend, 
                title=marker_by, 
                loc="lower left", 
                frameon=True,
                fontsize=font_size,
                title_fontsize=font_size
            )
            if legend:
                legend.set_zorder(1000)


# Convenience functions for backward compatibility
def build_network(*args, **kwargs):
    """Convenience function for building networks."""
    return NetworkBuilder.build_network(*args, **kwargs)


def plot_network(
    G: nx.Graph,
    pos: Dict,
    metric: str = "correlation",
    metadata: Optional[pd.DataFrame] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Convenience function for plotting networks."""
    plotter = NetworkPlotter(G, pos, metric, metadata)
    return plotter.plot(**kwargs)


def check_metadata_column(metadata: pd.DataFrame, column: str):
    """
    Utility function to check the type and values of a metadata column.
    Useful for debugging legend issues.
    
    Args:
        metadata: DataFrame containing metadata
        column: Column name to check
    """
    if column not in metadata.columns:
        print(f"❌ Column '{column}' not found in metadata")
        print(f"Available columns: {list(metadata.columns)}")
        return
    
    series = metadata[column]
    print(f"Column: {column}")
    print(f"  Data type: {series.dtype}")
    print(f"  Is numeric: {pd.api.types.is_numeric_dtype(series)}")
    print(f"  Unique values: {series.nunique()}")
    print(f"  Missing values: {series.isna().sum()}")
    print(f"  Sample values: {list(series.dropna().unique()[:5])}")
    
    if not pd.api.types.is_numeric_dtype(series):
        print(f"  → Will create LEGEND (categorical)")
    else:
        print(f"  → Will create COLORBAR (continuous)")