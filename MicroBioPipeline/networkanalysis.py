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

from .figuresetting import get_font_sizes
from .figuresetting import get_color_dict, pastelize_cmap

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
        edge_vmin: float = -1.0,
        edge_vmax: float = 1.0,
        figsize: Tuple[int, int] = (10, 10),
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        edge_width_scale: float = 1.0,
        edge_alpha: float = 0.8,
        node_alpha: float = 0.9,
        node_linewidth: float = 1.0,
        legend: bool = True,
        label_font_size: int = 10,
        legend_font_size: int = 10,
        title_font_size: int = 12,
        colorbar_font_size: int = 10,
        show: bool = True,
        default_equal_node_size: bool = True,
        default_node_size: float = 600,
        filename: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot network with customizable styling.
        
        Args:
            color_by: Metadata column for node colors
            size_by: Metadata column for node sizes
            marker_by: Metadata column for node shapes
            default_node_shape: Default marker shape
            cmap: Colormap for edges and continuous node colors
            edge_vmin: Minimum edge color value
            edge_vmax: Maximum edge color value
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
        self._draw_edges(ax, cmap, edge_vmin, edge_vmax, edge_width_scale, edge_alpha)
        
        # Draw nodes
        self._draw_nodes(
            ax, node_colors, node_sizes, node_markers,
            marker_by, node_alpha, node_linewidth
        )
        
        # Draw labels
        nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=label_font_size)
        
        # Add edge colorbar
        edge_sm = self._create_edge_scalar_mappable(cmap, edge_vmin, edge_vmax)
        cbar = fig.colorbar(edge_sm, ax=ax, pad=0.1)
        cbar.set_label(f"{self.metric.capitalize()}", fontsize=colorbar_font_size)
        cbar.ax.tick_params(labelsize=colorbar_font_size)
        
        # Add node colorbar if continuous coloring
        if color_sm is not None:
            cbar_nodes = fig.colorbar(color_sm, ax=ax, pad=0.02)
            cbar_nodes.set_label(color_by if color_by else "Value", fontsize=colorbar_font_size)
            cbar_nodes.ax.tick_params(labelsize=colorbar_font_size)
        
        # Add legends
        self._add_legends(
            legend,
            ax, color_legend, marker_legend, 
            color_by, marker_by, legend_font_size
        )

        ax.axis('off')

        if title is not None:
            fig.suptitle(title, fontsize=title_font_size)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        if filename is not None:
            fig.tight_layout()
            fig.savefig(filename, bbox_inches='tight', dpi=600, transparent=False)
        
        return fig, ax
    
    def _draw_edges(
        self, 
        ax: plt.Axes, 
        cmap: Union[str, mpl.colors.Colormap],
        edge_vmin: float,
        edge_vmax: float,
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
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
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
        cmap: Union[str, mpl.colors.Colormap],
        edge_vmin: float = -1.0,
        edge_vmax: float = 1.0
    ) -> mpl.cm.ScalarMappable:
        """Create scalar mappable for edge colorbar."""
        edge_norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_cmap = plt.cm.get_cmap(cmap)
        sm = mpl.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
        sm.set_array([])
        return sm
    
    def _add_legends(
        self,
        legend: bool,
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
        
        if legend:
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
        else:
            ax.legend_.remove() if ax.legend_ else None


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



# --------------------------------------------------------------------------------------------------------------
# LRG-based community detection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigh
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

class LRGCommunityDetector:
    def __init__(self, G, use_weights=True):
        self.G = G
        self.N = len(G)
        self.use_weights = use_weights
        if use_weights and nx.is_weighted(G):
            self.L = nx.laplacian_matrix(G, weight='weight').toarray()
        else:
            self.L = nx.laplacian_matrix(G).toarray()
        self.eigenvalues, self.eigenvectors = eigh(self.L)
        self.tau_range = None
        self.entropy = None
        self.susceptibility = None
        self.tau_peaks = None
        
    def compute_density_matrix(self, tau):
        exp_tau_lambda = np.exp(-tau * self.eigenvalues)
        Z = np.sum(exp_tau_lambda)
        # Compute eigenvalues of density matrix
        eigenvalues_rho = exp_tau_lambda / Z
        # Reconstruct density matrix
        rho = self.eigenvectors @ np.diag(eigenvalues_rho) @ self.eigenvectors.T
        return rho, eigenvalues_rho

    def compute_entropy(self, tau):
        # Compute von Neumann entropy
        _, eigenvalues_rho = self.compute_density_matrix(tau)
        # Avoid log(0) by filtering out zero eigenvalues
        nonzero_eigs = eigenvalues_rho[eigenvalues_rho > 1e-15]
        # Normalized von Neumann entropy
        S = - np.sum(nonzero_eigs * np.log10(nonzero_eigs)) / np.log10(self.N)
        return S

    def compute_susceptibility(self, tau_range=None, n_points=1000):
        # Determine tau range based on eigenvalues if not provided
        self.lambda_max = self.eigenvalues[-1]
        self.lambda_gap = self.eigenvalues[1] if self.eigenvalues[0] < 1e-10 else self.eigenvalues[0]
        # Set default tau range
        if tau_range is None:
            tau_min = 1.0 / self.lambda_max
            tau_max = 1.0 / self.lambda_gap
        else:
            tau_min, tau_max = tau_range        
        # Logarithmically spaced tau values
        self.tau_range = np.logspace(np.log10(tau_min), np.log10(tau_max), n_points)
        # Compute entropy for each tau
        self.entropy = np.array([self.compute_entropy(tau) for tau in self.tau_range])
        # Compute susceptibility as negative gradient of entropy w.r.t. log10(tau)
        log_tau = np.log10(self.tau_range)
        # Numerical gradient
        self.susceptibility = - np.gradient(self.entropy, log_tau)
        return self.tau_range, self.entropy, self.susceptibility

    def find_characteristic_scales(self, prominence=0.0):
        from scipy.signal import find_peaks
        # Ensure susceptibility is computed
        if self.susceptibility is None:
            self.compute_susceptibility()
        # Find peaks in susceptibility
        peaks, properties = find_peaks(self.susceptibility, prominence=prominence)
        # Compute derivative of susceptibility
        dchi_dlogtau = np.gradient(self.susceptibility, np.log10(self.tau_range))
        # Get tau values at peaks ensuring null derivative
        valid_peaks = []
        for peak in peaks:
            valid_peaks.append(peak)
        self.tau_peaks = self.tau_range[valid_peaks]
        return self.tau_peaks

    def compute_communicability(self, tau):
        # Compute communicability matrix K = exp(-tau * L)
        exp_tau_lambda = np.exp(-tau * self.eigenvalues)
        # Reconstruct K
        K = self.eigenvectors @ np.diag(exp_tau_lambda) @ self.eigenvectors.T
        return K

    def compute_distance_matrix(self, tau):
        # Compute distance matrix D from communicability
        K = self.compute_communicability(tau)
        # Avoid division by zero on diagonal
        D = np.zeros_like(K)
        mask = ~np.eye(self.N, dtype=bool)
        D[mask] = 1.0 / K[mask]
        return D

    # def compute_partition_stability(self, Z, tau):
    #     delta = np.sort(Z[:, 2])[::-1]

    #     # Extend boundaries
    #     delta_extended = np.concatenate([[delta[0] / 10], delta, [delta[-1] * 10]])
    #     log_delta = np.log10(delta_extended + 1e-15)

    #     # Compute differences
    #     log_diff = log_delta[:-1] - log_delta[1:]

    #     # Norm (first and last *real* values)
    #     norm = log_delta[1] - log_delta[-2]
    #     if abs(norm) < 1e-10:
    #         norm = 1.0

    #     N = 1.0 / norm
    #     psi = N * log_diff

    #     # Only real splits for psi (exclude extended ends)
    #     psi = psi[1:-1]

    #     # Index of optimal gap
    #     optimal_n_clusters = np.argmax(psi) + 2

    #     print(f"Optimal index: {optimal_n_clusters}, Psi value: \n{psi}")

    #     return psi, optimal_n_clusters
    
    ########################################################
    def compute_partition_stability(self, Z, tau=None, method='combined'):
        """
        Compute partition stability to find optimal number of clusters.
        
        Parameters:
        -----------
        Z : ndarray
            Linkage matrix from hierarchical clustering
        tau : float, optional
            Threshold parameter (currently unused, can be for future extensions)
        method : str
            'log_gap', 'acceleration', 'combined', or 'all'
        
        Returns:
        --------
        dict with stability metrics and optimal cluster number
        """
        
        # Get merge distances (heights)
        delta = np.sort(Z[:, 2])[::-1]  # Descending order
        n_samples = Z.shape[0] + 1
        n_possible_clusters = len(delta)
        
        results = {}
        
        # ===== Method 1: Improved Log-Gap (your original approach) =====
        if method in ['log_gap', 'combined', 'all']:
            # Use log1p for numerical stability instead of log10(delta + eps)
            log_delta = np.log1p(delta)
            
            # Compute consecutive differences (gap sizes)
            log_diff = log_delta[:-1] - log_delta[1:]
            
            # Normalize by total range
            norm = log_delta[0] - log_delta[-1]
            if abs(norm) < 1e-10:
                norm = 1.0
            
            psi_log = log_diff / norm
            optimal_log = np.argmax(psi_log) + 2
            
            results['psi_log'] = psi_log
            results['optimal_log'] = optimal_log
        
        # ===== Method 2: Acceleration (Second Derivative) =====
        if method in ['acceleration', 'combined', 'all']:
            # Compute first derivative (rate of change)
            first_diff = np.diff(delta)
            
            # Compute second derivative (acceleration)
            second_diff = np.diff(first_diff)
            
            # Normalize
            if np.max(np.abs(second_diff)) > 1e-10:
                psi_accel = second_diff / np. max(np.abs(second_diff))
            else:
                psi_accel = second_diff
            
            # Find maximum jump (most negative acceleration = biggest elbow)
            optimal_accel = np.argmin(psi_accel) + 2
            
            results['psi_accel'] = psi_accel
            results['optimal_accel'] = optimal_accel
        
        # ===== Method 3: Relative Gap =====
        if method in ['combined', 'all']:
            # Compute relative gap:  (delta[i] - delta[i+1]) / delta[i+1]
            relative_gap = np.zeros(len(delta) - 1)
            for i in range(len(delta) - 1):
                if delta[i+1] > 1e-10:
                    relative_gap[i] = (delta[i] - delta[i+1]) / delta[i+1]
                else:
                    relative_gap[i] = 0
            
            optimal_relative = np.argmax(relative_gap) + 2
            
            results['psi_relative'] = relative_gap
            results['optimal_relative'] = optimal_relative
        
        # ===== Method 4: Calinski-Harabasz inspired metric =====
        if method == 'all':
            # Ratio of between-cluster to within-cluster variance proxy
            psi_ch = np.zeros(len(delta))
            for i in range(len(delta)):
                n_clusters = i + 2
                if n_clusters < n_samples:
                    # Higher delta = better separation at this level
                    # Normalized by number of clusters
                    psi_ch[i] = delta[i] * (n_samples - n_clusters) / (n_clusters - 1)
            
            optimal_ch = np. argmax(psi_ch) + 2
            
            results['psi_ch'] = psi_ch
            results['optimal_ch'] = optimal_ch
        
        # ===== Combined Score =====
        if method == 'combined':
            # Normalize all metrics to [0, 1]
            psi_log_norm = (psi_log - psi_log.min()) / (psi_log.max() - psi_log.min() + 1e-10)
            
            # Pad acceleration to match length
            psi_accel_padded = np.zeros(len(psi_log_norm))
            psi_accel_norm = (psi_accel - psi_accel.min()) / (psi_accel.max() - psi_accel.min() + 1e-10)
            psi_accel_padded[: len(psi_accel_norm)] = psi_accel_norm
            
            psi_rel_norm = (relative_gap - relative_gap.min()) / (relative_gap.max() - relative_gap.min() + 1e-10)
            
            # Combined weighted score
            psi_combined = (0.4 * psi_log_norm + 
                        0.4 * psi_rel_norm + 
                        0.2 * psi_accel_padded[: len(psi_rel_norm)])
            
            optimal_combined = np.argmax(psi_combined) + 2
            
            results['psi_combined'] = psi_combined
            results['optimal_combined'] = optimal_combined
        
        # ===== Return results based on method =====
        if method == 'log_gap':
            return results['psi_log'], results['optimal_log']
        elif method == 'acceleration': 
            return results['psi_accel'], results['optimal_accel']
        elif method == 'combined':
            optimal_n_clusters = results['optimal_combined']
        else:  # 'all'
            # Vote among methods
            votes = [results. get(f'optimal_{m}', 0) 
                    for m in ['log', 'accel', 'relative', 'ch']]
            optimal_n_clusters = int(np.median(votes))
            results['optimal_vote'] = optimal_n_clusters
        
        # Print diagnostics
        print(f"Optimal number of clusters: {optimal_n_clusters}")
        if method in ['combined', 'all']:
            print(f"  - Log-gap suggests:  {results. get('optimal_log', 'N/A')}")
            print(f"  - Acceleration suggests: {results.get('optimal_accel', 'N/A')}")
            print(f"  - Relative gap suggests:  {results.get('optimal_relative', 'N/A')}")
            if method == 'all':
                print(f"  - CH-inspired suggests: {results.get('optimal_ch', 'N/A')}")
        
        return results, optimal_n_clusters


    def visualize_stability(self,
                            Z,
                            results,
                            method='combined',
                            figsize=(14, 10),
                            filename=None):
        """
        Visualize the partition stability metrics.
        """        
        n_clusters_range = np.arange(2, len(Z[: , 2]) + 2)

        font_size = get_font_sizes(figsize[0]/1.5, figsize[1]/1.5, "in")
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Dendrogram with cut line
        from scipy.cluster.hierarchy import dendrogram
        ax = axes[0, 0]
        dendrogram(Z, ax=ax, no_labels=True)
        
        if method == 'combined':
            optimal = results['optimal_combined']
            threshold = Z[-(optimal-1), 2]
        else:
            optimal = results. get('optimal_vote', results. get('optimal_log', 2))
            threshold = Z[-(optimal-1), 2]
        
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                   label=f'Cut at k={optimal}')
        ax.set_title('Dendrogram with Optimal Cut', fontsize=font_size['title'], pad=20)
        ax.set_xlabel('Samples', fontsize=font_size['label'])
        ax.set_ylabel('Distance', fontsize=font_size['label'])
        ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
        ax.legend(fontsize=font_size['legend'], loc='upper right')
        
        # Plot 2: Merge distances
        ax = axes[0, 1]
        delta = np.sort(Z[:, 2])[::-1]
        ax.plot(n_clusters_range, delta, 'o-', linewidth=2)
        ax.axvline(x=optimal, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Number of Clusters', fontsize=font_size['label'])
        ax.set_ylabel('Merge Distance (Height)', fontsize=font_size['label'])
        ax.set_title('Linkage Distances', fontsize=font_size['title'], pad=20)
        ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Stability metrics
        ax = axes[1, 0]
        if 'psi_combined' in results:
            psi = results['psi_combined']
            ax.plot(n_clusters_range[: len(psi)], psi, 'o-', linewidth=2, label='Combined Psi')
        elif 'psi_log' in results:
            psi = results['psi_log']
            ax.plot(n_clusters_range[:len(psi)], psi, 'o-', linewidth=2, label='Log-gap Psi')
        
        ax.axvline(x=optimal, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal}')
        ax.set_xlabel('Number of Clusters', fontsize=font_size['label'])
        ax.set_ylabel('Stability Score (Psi)', fontsize=font_size['label'])
        ax.set_title('Partition Stability', fontsize=font_size['title'], pad=20)
        ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
        ax.legend(fontsize=font_size['legend'])
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Multiple metrics comparison
        ax = axes[1, 1]
        if method in ['combined', 'all']: 
            if 'psi_log' in results:
                psi_log = results['psi_log']
                psi_log_norm = (psi_log - psi_log.min()) / (psi_log.max() - psi_log.min() + 1e-10)
                ax.plot(n_clusters_range[:len(psi_log_norm)], psi_log_norm, 
                    'o-', label='Log-gap', alpha=0.7)
            
            if 'psi_relative' in results:
                psi_rel = results['psi_relative']
                psi_rel_norm = (psi_rel - psi_rel.min()) / (psi_rel.max() - psi_rel.min() + 1e-10)
                ax.plot(n_clusters_range[:len(psi_rel_norm)], psi_rel_norm, 
                    's-', label='Relative gap', alpha=0.7)
            
            ax.axvline(x=optimal, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Number of Clusters', fontsize=font_size['label'])
            ax.set_ylabel('Normalized Score', fontsize=font_size['label'])
            ax.set_title('Comparison of Metrics', fontsize=font_size['title'], pad=20)
            ax.legend(fontsize=font_size['legend'])
            ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', dpi=600, transparent=False)

    ########################################################

    def detect_communities_at_scale(self, tau, method='hierarchical', n_clusters=None, return_linkage=False):
        # Compute distance matrix
        D = self.compute_distance_matrix(tau)
        # Hierarchical clustering
        if method == 'hierarchical':
            D_condensed = squareform(D, checks=False)
            Z = linkage(D_condensed, method='average')
            if n_clusters is None:
                psi, n_clusters = self.compute_partition_stability(Z, tau)
            labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
            # Compute silhouette score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(D, labels, metric='precomputed')
            else:
                score = 0.0
            # Return labels and optionally linkage matrix
            if return_linkage:
                return labels, score, Z
            else:
                return labels, score
        
        # K-means clustering
        elif method == 'kmeans':
            K = self.compute_communicability(tau)
            if n_clusters is None:
                n_clusters = self._estimate_n_clusters(K)
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(K)
            # Compute silhouette score
            score = silhouette_score(D, labels, metric='precomputed')
            
            return labels, score
        
        # Spectral clustering
        elif method == 'spectral':
            if n_clusters is None:
                n_clusters = self._estimate_n_clusters(self.compute_communicability(tau))
            # Perform spectral clustering
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
            K = self.compute_communicability(tau)
            labels = spectral.fit_predict(K)
            # Compute silhouette score
            score = silhouette_score(D, labels, metric='precomputed')
            
            return labels, score
        
        else:
            raise ValueError(f"Unknown method: {method}")

    def _estimate_n_clusters(self, X, max_clusters=100):
        scores = []
        K_range = range(2, min(max_clusters + 1, self.N))
        # Evaluate silhouette scores for different k
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)
        # Select k with highest silhouette score
        optimal_k = K_range[np.argmax(scores)]

        return optimal_k

    def detect_metastable_nodes(self, tau_values, method='hierarchical'):
        n_scales = len(tau_values)
        labels_matrix = np.zeros((n_scales, self.N), dtype=int)
        # Detect communities at each scale
        for i, tau in enumerate(tau_values):
            labels, _ = self.detect_communities_at_scale(tau, method=method)
            labels_matrix[i, :] = labels
        # Compute stability scores
        stability_scores = np.zeros(self.N)
        for node in range(self.N):
            transitions = np.sum(labels_matrix[:-1, node] != labels_matrix[1:, node])
            stability_scores[node] = 1.0 - (transitions / (n_scales - 1))
        # Identify metastable nodes
        threshold = 0.5
        metastable_indices = np.where(stability_scores < threshold)[0]

        return metastable_indices, stability_scores, labels_matrix

# Visualization functions
def plot_lrg_analysis(detector,
                      pos=None,
                      G=None,
                      figsize=(15, 10),
                      filenmame=None):
    """Plot LRG analysis results including entropy, susceptibility, spectrum, and network."""
    # Ensure susceptibility is computed
    if detector.susceptibility is None:
        detector.compute_susceptibility()
    
    # Find characteristic scales
    tau_peaks = detector.find_characteristic_scales()
    
    # Compute specific heat (second derivative)
    log_tau = np.log10(detector.tau_range)
    d_specific_heat = np.gradient(np.gradient(detector.entropy, log_tau), log_tau)
    
    font_size = get_font_sizes(figsize[0], figsize[1], "in")
    
    # Entropy
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.semilogx(detector.tau_range, detector.entropy, 'b')
    ax.set_xlabel(r'Diffusion time $\tau$', fontsize=font_size['axes_label'])
    ax.set_ylabel(r'Entropy $S(\tau)$', fontsize=font_size['axes_label'])
    ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
    ax.set_title('Network Entropy', fontsize=font_size['title'], pad=20)
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    if filenmame is not None:
        fig.savefig('entropy_' + filenmame, bbox_inches='tight', dpi=600, transparent=False)    

    # Susceptibility & Specific Heat
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ln1 = ax.semilogx(detector.tau_range, detector.susceptibility, 'r-', label='Susceptibility')
    ax.scatter(tau_peaks, detector.susceptibility[np.isin(detector.tau_range, tau_peaks)], 
               s=150, c='red', marker='*', zorder=5, label='Peaks', edgecolors='black', linewidths=1.5)
    ax.set_xlabel(r'Diffusion time $\tau$', fontsize=font_size['axes_label'])
    ax.set_ylabel(r'Susceptibility $C(\tau)$', fontsize=font_size['axes_label'], color='red')
    ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
    ax.set_title('Entropy and Susceptibility', fontsize=font_size['title'], pad=20)
    ax.grid(True, alpha=0.5)

    for tau_peak in tau_peaks:
        ax.axvline(x=tau_peak, color='gray', linestyle='--', alpha=0.7)
    
    # ax2 = ax.twinx()
    # ln2 = ax2.semilogx(detector.tau_range[2:-2], d_specific_heat[2:-2], 'g--', label='Derivative of Susceptibility')
    # ax2.axhline(0, color='gray', linestyle='--', alpha=1)
    # ax2.set_ylabel(r'Derivative of Susceptibility', fontsize=font_size['axes_label'], color='green')
    # ax2.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(handles1+handles2, labels1+labels2, loc='best', fontsize=font_size['legend'])

    ln2 = ax.semilogx(detector.tau_range, detector.entropy, 'b--', label='Entropy S($\\tau$)')
    ax.axhline(0, color='gray', linestyle='--', alpha=1)
    ax.set_ylabel(r'Entropy', fontsize=font_size['axes_label'], color='blue')
    ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
    handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax.get_legend_handles_labels()
    ax.legend(handles1, labels1, loc='best', fontsize=font_size['legend'])
    
    plt.tight_layout()
    if filenmame is not None:
        fig.savefig('susceptibility_' + filenmame, bbox_inches='tight', dpi=600, transparent=False)

    # Laplacian Spectrum
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # ax.semilogy(detector.eigenvalues, 'o-', markersize=8, alpha=0.8)
    ax.plot(detector.eigenvalues, 'o-', markersize=8, alpha=0.8)
    ax.set_xlabel('Index', fontsize=font_size['axes_label'])
    ax.set_ylabel(r'Eigenvalue $\lambda$', fontsize=font_size['axes_label'])
    ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
    ax.set_title(f'Laplacian Spectrum\n' + r'$\lambda_{max}$' + f' = {detector.eigenvalues.max():.2g}' +
                 r' - $\lambda_{gap}$' + f' = {detector.lambda_gap:.2g}',
                 fontsize=font_size['title'], pad=20)
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    if filenmame is not None:
        fig.savefig('spectrum_' + filenmame, bbox_inches='tight', dpi=600, transparent=False)
    
    print(f"\nFound {len(tau_peaks)} characteristic scales:")
    for i, tau in enumerate(tau_peaks):
        print(f"  τ*_{i+1} = {tau:.4g}")
    
    return tau_peaks

# Plot dendrogram and community graph
def plot_dendrogram(Z,
                    G,
                    pos,
                    labels, 
                    tau=None,
                    score=None,
                    figsize=(12, 6),
                    filename=None):
    """
    Plots hierarchical clustering dendrogram (log-scale distance) and the community graph. 
    """

    font_size = get_font_sizes(figsize[0], figsize[1], "in")
    
    # Dendrogram
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    dn = dendrogram(Z, ax=ax, color_threshold=Z[-len(np.unique(labels))+1, 2])
    
    if score is not None:
        ax.axhline(score, color='red', linestyle='--', label=r'$\Psi$ score')
    # Get the leaf order from dendrogram
    leaf_order = dn['leaves']
    
    # Get the actual node labels from graph G
    # Convert G.nodes() to a list to allow indexing
    node_list = list(G.nodes())
    
    # Map leaf order indices to actual node labels from G
    node_labels = [str(node_list[i]) for i in leaf_order]
    
    ax.set_xlabel('Node Index', fontsize=font_size['axes_label'])
    ax.set_ylabel('Distance (log scale)', fontsize=font_size['axes_label'])
    
    # Get x lim
    xlim = ax.get_xlim()
    ax.set_xticks(np.linspace(xlim[0], xlim[1], len(node_labels)))
    ax.set_xticklabels(node_labels, fontsize=font_size['ticks_label'], rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=font_size['ticks_label'])
    ax.set_title(f'Hierarchical Dendrogram' + (f'\nτ={tau:.3g}' if tau is not None else ''), 
                 fontsize=font_size['title'], pad=20)
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=600, bbox_inches='tight', transparent=False)



# Community graph
def plot_communities(G,
                     pos,
                     df_info,
                     labels,
                     tau=None,
                     figsize=(12, 6),
                     cmap='viridis',
                     edge_vmin=-1,
                     edge_vmax=1,
                     edgescale=5,
                     filename=None
                     ):
    """
    Docstring for plot_communities
    
    :param G: Description
    : param pos: Description
    :param info: Description
    :param labels: Description
    :param tau:  Description
    :param figsize:  Description
    """

    font_size = get_font_sizes(figsize[0], figsize[1], "in")
    # Add to info dataframe the community labels for nodes present in G
    df_info['community'] = None  # Initialize with None for all rows
    nodes = list(G.nodes())
    df_info. loc[nodes, 'community'] = labels

    fig, ax = plot_network(
        G, pos, 
        metadata=df_info,
        color_by='community',
        edge_width_scale=edgescale,
        figsize=(figsize[0], figsize[1]),
        title=f"Network Plot - Communities (τ={tau:.3g})" if tau is not None else "Network Plot - Communities",
        legend=False,
        legend_font_size=font_size["legend"]//2,
        colorbar_font_size=font_size["cbar_ticks"],
        label_font_size=font_size["ticks_label"]//3,
        title_font_size=font_size["title"],
        cmap=cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        default_equal_node_size=True,
        default_node_size=300,
        filename=filename
    )


# Plot metastable nodes
def plot_metastable_nodes(G,
                          pos,
                          df_info,
                          metastable_indices,
                          stability_scores,
                          figsize=(10, 8),
                          cmap='viridis',
                          edge_vmin=-1,
                          edge_vmax=1,
                          edgescale=5,
                          filename=None
                          ):
    
    font_size = get_font_sizes(figsize[0], figsize[1], "in")

    # Add metastability information to df_info
    df_info['is_metastable'] = False
    df_info['stability_score'] = None
    
    nodes = list(G.nodes())
    
    # Mark metastable nodes
    metastable_nodes = [nodes[i] for i in metastable_indices if i < len(nodes)]
    df_info.loc[metastable_nodes, 'is_metastable'] = True
    
    # Assign stability scores to all nodes in G
    df_info.loc[nodes, 'stability_score'] = stability_scores[: len(nodes)]
    
    fig, ax = plot_network(
        G, pos, 
        metadata=df_info,
        color_by='is_metastable',  # or 'stability_score' for continuous coloring
        edge_width_scale=edgescale,
        figsize=figsize,
        title='Metastable Nodes (unstable across scales)',
        legend=False,
        legend_font_size=font_size["legend"]//2,
        colorbar_font_size=font_size["cbar_ticks"],
        label_font_size=font_size["ticks_label"]//3,
        title_font_size=font_size["title"],
        cmap=cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        default_equal_node_size=False,  # Allow variable sizes
        default_node_size=300,
        filename=filename
    )

# Plot Sankey diagram for community evolution
def plot_sankey_diagram(labels_df,
                        tau_values,
                        filename=None):    
    N, n_scales = labels_df.shape

    print(labels_df)
    
    sources = []
    targets = []
    values = []
    labels = []
    offset = 0
    node_map = {}
    for i, scale in enumerate(labels_df.columns):
        unique_communities = np.unique(labels_df.iloc[:, i])
        for comm in unique_communities:
            node_id = offset + comm
            node_map[(i, comm)] = node_id
            # FIRST SCALE: use index labels as node labels
            if i == 0:
                # For each occurrence of comm in column i, label with index
                matching_indices = labels_df.index[labels_df.iloc[:, i] == comm]
                for idx in matching_indices:
                    labels.append(str(idx))
            else:
                labels.append(f"C{comm}")
        offset += len(unique_communities)
    
    # Adjust for node id mapping and label order in multi-index
    sources = []
    targets = []
    values = []
    for scale in range(n_scales - 1):
        for node in range(N):
            src_comm = labels_df.iloc[node, scale]
            tgt_comm = labels_df.iloc[node, scale+1]
            src_id = node_map[(scale, src_comm)]
            tgt_id = node_map[(scale + 1, tgt_comm)]
            try:
                idx = list(zip(sources, targets)).index((src_id, tgt_id))
                values[idx] += 1
            except ValueError:
                sources.append(src_id)
                targets.append(tgt_id)
                values.append(1)

    # If you want only the first scale nodes to get the 'labels_df.index' label,
    # and later-scale communities just get 'C{comm}', you'll need to shift offset for each index label, possibly
    # Use this for simple cases and adapt as needed for complex community merges

    font_size = get_font_sizes(10, 6, "in")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=50,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    )])

    h = N * 30
    w = max(1000, n_scales * 300)

    fig.update_layout(
        title_text="Community Evolution Across Scales",
        font_size=font_size['title'],
        height=h,
        width=w,
    )

    if filename:
        fig.write_image(filename, scale=2)
