# Annotated Network Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from typing import Optional, Dict, Tuple, Union, Literal, List, Sequence, Callable


def create_network_from_data(
    data: Union[pd.DataFrame, nx.Graph, nx.DiGraph],
    graph_type: Literal['undirected', 'directed'] = 'undirected',
    threshold: Optional[float] = None,
    absolute_threshold: bool = False,
    keep_negative: bool = True,
    self_loops: bool = False,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Create a NetworkX graph from various input data formats.
    
    Parameters
    ----------
    data : pd.DataFrame, nx.Graph, or nx.DiGraph
        Input data. Can be:
        - Adjacency matrix (square DataFrame with node names as index/columns)
        - Edge list (DataFrame with 'source', 'target', and optional 'weight' columns)
        - Existing NetworkX graph (will be returned as-is or converted)
    graph_type : str, default 'undirected'
        Type of graph to create: 'undirected' or 'directed'.
    threshold : float, optional
        Minimum edge weight to include. Edges below this threshold are excluded.
    absolute_threshold : bool, default False
        If True, use absolute value of weights for threshold comparison.
    keep_negative : bool, default True
        If True, keep negative edge weights; if False, exclude them.
    self_loops : bool, default False
        If True, include self-loops (diagonal in adjacency matrix).
    
    Returns
    -------
    G : nx.Graph or nx.DiGraph
        NetworkX graph object.
    
    Examples
    --------
    >>> # From adjacency matrix
    >>> adj_matrix = pd.DataFrame(...)
    >>> G = create_network_from_data(adj_matrix, threshold=0.5)
    
    >>> # From edge list
    >>> edges = pd.DataFrame({'source': [...], 'target': [...], 'weight': [...]})
    >>> G = create_network_from_data(edges, graph_type='directed')
    
    >>> # From existing graph, convert to directed
    >>> G_directed = create_network_from_data(G_undirected, graph_type='directed')
    """
    
    # If already a NetworkX graph
    if isinstance(data, (nx.Graph, nx.DiGraph)):
        if graph_type == 'directed' and isinstance(data, nx.Graph):
            return data.to_directed()
        elif graph_type == 'undirected' and isinstance(data, nx.DiGraph):
            return data.to_undirected()
        return data.copy()
    
    # Create graph based on type
    if graph_type == 'directed':
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Check if data is an adjacency matrix (square DataFrame)
    if isinstance(data, pd.DataFrame):
        if data.shape[0] == data.shape[1] and all(data.index == data.columns):
            # Adjacency matrix
            nodes = data.index.tolist()
            G.add_nodes_from(nodes)
            
            for i, source in enumerate(data.index):
                for j, target in enumerate(data.columns):
                    # Skip self-loops if not wanted
                    if not self_loops and i == j:
                        continue
                    
                    weight = data.iloc[i, j]
                    
                    # Skip NaN values
                    if pd.isna(weight):
                        continue
                    
                    # Apply threshold
                    if threshold is not None:
                        weight_to_check = abs(weight) if absolute_threshold else weight
                        if weight_to_check < threshold:
                            continue
                    
                    # Skip negative weights if requested
                    if not keep_negative and weight < 0:
                        continue
                    
                    # For undirected graphs, only add edge once
                    if graph_type == 'undirected' and i > j:
                        continue
                    
                    G.add_edge(source, target, weight=weight)
        
        else:
            # Edge list format
            # Expected columns: 'source', 'target', and optional 'weight'
            required_cols = ['source', 'target']
            if not all(col in data.columns for col in required_cols):
                raise ValueError("Edge list DataFrame must have 'source' and 'target' columns")
            
            # Add nodes
            nodes = pd.unique(data[['source', 'target']].values.ravel())
            G.add_nodes_from(nodes)
            
            # Add edges
            for idx, row in data.iterrows():
                source = row['source']
                target = row['target']
                weight = row.get('weight', 1.0)
                
                # Skip self-loops if not wanted
                if not self_loops and source == target:
                    continue
                
                # Skip NaN weights
                if pd.isna(weight):
                    continue
                
                # Apply threshold
                if threshold is not None:
                    weight_to_check = abs(weight) if absolute_threshold else weight
                    if weight_to_check < threshold:
                        continue
                
                # Skip negative weights if requested
                if not keep_negative and weight < 0:
                    continue
                
                # Add edge with attributes
                edge_attrs = {k: v for k, v in row.items() if k not in ['source', 'target']}
                G.add_edge(source, target, **edge_attrs)
    
    return G


def plot_annotated_network(
    G: Union[nx.Graph, nx.DiGraph, pd.DataFrame],
    node_annotations: Optional[pd.DataFrame] = None,
    edge_annotations: Optional[pd.DataFrame] = None,
    # Graph creation parameters (if G is DataFrame)
    graph_type: Literal['undirected', 'directed'] = 'undirected',
    threshold: Optional[float] = None,
    absolute_threshold: bool = False,
    keep_negative: bool = True,
    self_loops: bool = False,
    # Figure parameters
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 10),
    title: str = 'Network Graph',
    # Layout parameters
    layout: Union[str, Dict, Callable] = 'spring',
    layout_kws: Optional[Dict] = None,
    # Node parameters
    node_size_col: Optional[str] = None,
    node_size_map: Optional[Dict] = None,
    node_size_scale: float = 300,
    node_color_col: Optional[str] = None,
    node_palette: Optional[Dict[str, str]] = None,
    node_cmap: Union[str, LinearSegmentedColormap] = 'viridis',
    node_vmin: Optional[float] = None,
    node_vmax: Optional[float] = None,
    node_shape_col: Optional[str] = None,
    node_shape_map: Optional[Dict[str, str]] = None,
    default_node_size: float = 300,
    default_node_color: str = 'lightblue',
    default_node_shape: str = 'o',
    node_alpha: float = 1.0,
    node_edgecolors: str = 'black',
    node_linewidths: float = 1.0,
    # Node label parameters
    show_node_labels: bool = True,
    node_labels: Optional[Dict] = None,
    node_label_size: float = 10,
    node_label_color: str = 'black',
    node_label_position: Literal['center', 'above', 'below'] = 'center',
    node_label_offset: float = 0.05,
    # Edge parameters
    edge_width_col: Optional[str] = None,
    edge_width_scale: float = 2.0,
    edge_color_col: Optional[str] = None,
    edge_palette: Optional[Dict] = None,
    edge_cmap: Union[str, LinearSegmentedColormap] = 'coolwarm',
    edge_vmin: Optional[float] = None,
    edge_vmax: Optional[float] = None,
    default_edge_width: float = 1.0,
    default_edge_color: str = 'gray',
    edge_alpha: float = 0.7,
    edge_style: str = 'solid',
    # Edge label parameters
    show_edge_labels: bool = False,
    edge_labels: Optional[Dict] = None,
    edge_label_col: Optional[str] = None,
    edge_label_size: float = 8,
    edge_label_color: str = 'black',
    # Arrow parameters (for directed graphs)
    arrows: bool = True,
    arrowsize: int = 10,
    arrowstyle: str = '->',
    connectionstyle: str = 'arc3,rad=0.1',
    # Legend parameters
    show_node_legend: bool = True,
    show_edge_legend: bool = False,
    node_legend_title: str = 'Nodes',
    edge_legend_title: str = 'Edges',
    legend_position: Literal['right', 'left', 'top', 'bottom'] = 'right',
    legend_bbox: Tuple[float, float] = (1.02, 0.5),
    legend_fontsize: float = 10,
    # Colorbar parameters
    show_node_colorbar: bool = False,
    show_edge_colorbar: bool = False,
    node_colorbar_label: Optional[str] = None,
    edge_colorbar_label: Optional[str] = None,
    colorbar_orientation: Literal['vertical', 'horizontal'] = 'vertical',
    # Save parameters
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes, Union[nx.Graph, nx.DiGraph], Dict]:
    """
    Plot an annotated network graph with customizable node and edge properties.
    
    Parameters
    ----------
    G : nx.Graph, nx.DiGraph, or pd.DataFrame
        NetworkX graph or adjacency matrix/edge list to convert to graph.
    node_annotations : pd.DataFrame, optional
        DataFrame with node attributes (index should match node names).
        Columns can be used for node_size_col, node_color_col, node_shape_col.
    edge_annotations : pd.DataFrame, optional
        DataFrame with edge attributes. Must have 'source' and 'target' columns.
        Other columns can be used for edge_width_col, edge_color_col.
    graph_type : str, default 'undirected'
        Type of graph: 'undirected' or 'directed'.
    threshold : float, optional
        Minimum edge weight to include (when creating from DataFrame).
    absolute_threshold : bool, default False
        Use absolute values for threshold.
    keep_negative : bool, default True
        Keep negative edge weights.
    self_loops : bool, default False
        Include self-loops.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    figsize : tuple, default (12, 10)
        Figure size (width, height).
    title : str, default 'Network Graph'
        Plot title.
    layout : str, dict, or callable, default 'spring'
        Graph layout. Options:
        - 'spring': Force-directed layout
        - 'circular': Nodes in a circle
        - 'kamada_kawai': Force-directed with optimal distances
        - 'shell': Concentric circles
        - 'spectral': Based on graph Laplacian eigenvectors
        - dict: Pre-computed positions {node: (x, y)}
        - callable: Custom layout function
    layout_kws : dict, optional
        Additional arguments for layout algorithm.
    node_size_col : str, optional
        Column in node_annotations to use for node sizes.
    node_size_map : dict, optional
        Mapping from categories to sizes (if node_size_col is categorical).
    node_size_scale : float, default 300
        Scaling factor for node sizes.
    node_color_col : str, optional
        Column in node_annotations to use for node colors.
    node_palette : dict, optional
        Mapping from categories to colors (if node_color_col is categorical).
    node_cmap : str or colormap, default 'viridis'
        Colormap for continuous node colors.
    node_vmin : float, optional
        Minimum value for node color normalization.
    node_vmax : float, optional
        Maximum value for node color normalization.
    node_shape_col : str, optional
        Column in node_annotations to use for node shapes.
    node_shape_map : dict, optional
        Mapping from categories to shapes: 'o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+', 'x'.
    default_node_size : float, default 300
        Default node size.
    default_node_color : str, default 'lightblue'
        Default node color.
    default_node_shape : str, default 'o'
        Default node shape (matplotlib marker).
    node_alpha : float, default 1.0
        Node transparency.
    node_edgecolors : str, default 'black'
        Color of node borders.
    node_linewidths : float, default 1.0
        Width of node borders.
    show_node_labels : bool, default True
        Whether to show node labels.
    node_labels : dict, optional
        Custom node labels {node: label}.
    node_label_size : float, default 10
        Font size for node labels.
    node_label_color : str, default 'black'
        Color of node labels.
    node_label_position : str, default 'center'
        Label position: 'center', 'above', or 'below'.
    node_label_offset : float, default 0.05
        Offset for 'above' or 'below' label positions.
    edge_width_col : str, optional
        Column in edge_annotations to use for edge widths.
    edge_width_scale : float, default 2.0
        Scaling factor for edge widths.
    edge_color_col : str, optional
        Column in edge_annotations to use for edge colors.
    edge_palette : dict, optional
        Mapping from categories to colors (if edge_color_col is categorical).
    edge_cmap : str or colormap, default 'coolwarm'
        Colormap for continuous edge colors.
    edge_vmin : float, optional
        Minimum value for edge color normalization.
    edge_vmax : float, optional
        Maximum value for edge color normalization.
    default_edge_width : float, default 1.0
        Default edge width.
    default_edge_color : str, default 'gray'
        Default edge color.
    edge_alpha : float, default 0.7
        Edge transparency.
    edge_style : str, default 'solid'
        Edge line style: 'solid', 'dashed', 'dotted', 'dashdot'.
    show_edge_labels : bool, default False
        Whether to show edge labels.
    edge_labels : dict, optional
        Custom edge labels {(source, target): label}.
    edge_label_col : str, optional
        Column in edge_annotations to use for edge labels.
    edge_label_size : float, default 8
        Font size for edge labels.
    edge_label_color : str, default 'black'
        Color of edge labels.
    arrows : bool, default True
        Show arrows for directed graphs.
    arrowsize : int, default 10
        Size of arrows.
    arrowstyle : str, default '->'
        Style of arrows.
    connectionstyle : str, default 'arc3,rad=0.1'
        Style of edge connections.
    show_node_legend : bool, default True
        Show legend for node categories.
    show_edge_legend : bool, default False
        Show legend for edge categories.
    node_legend_title : str, default 'Nodes'
        Title for node legend.
    edge_legend_title : str, default 'Edges'
        Title for edge legend.
    legend_position : str, default 'right'
        Legend position.
    legend_bbox : tuple, default (1.02, 0.5)
        Legend bounding box anchor.
    legend_fontsize : float, default 10
        Legend font size.
    show_node_colorbar : bool, default False
        Show colorbar for continuous node colors.
    show_edge_colorbar : bool, default False
        Show colorbar for continuous edge colors.
    node_colorbar_label : str, optional
        Label for node colorbar.
    edge_colorbar_label : str, optional
        Label for edge colorbar.
    colorbar_orientation : str, default 'vertical'
        Colorbar orientation.
    save_path : str, optional
        Path to save figure.
    dpi : int, default 300
        DPI for saved figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    G : nx.Graph or nx.DiGraph
        The NetworkX graph object.
    info : dict
        Dictionary with plotting information (positions, node groups, edge groups).
    
    Examples
    --------
    >>> # From adjacency matrix with node annotations
    >>> G = create_network_from_data(adj_matrix, threshold=0.5)
    >>> fig, ax, G, info = plot_annotated_network(
    ...     G,
    ...     node_annotations=node_df,
    ...     node_size_col='degree',
    ...     node_color_col='community',
    ...     node_palette={'A': 'red', 'B': 'blue'},
    ...     layout='spring'
    ... )
    
    >>> # Directed network with edge weights
    >>> fig, ax, G, info = plot_annotated_network(
    ...     edge_list_df,
    ...     graph_type='directed',
    ...     edge_annotations=edge_df,
    ...     edge_width_col='weight',
    ...     edge_color_col='type',
    ...     arrows=True
    ... )
    """
    
    # Create graph if input is DataFrame
    if isinstance(G, pd.DataFrame):
        G = create_network_from_data(
            G,
            graph_type=graph_type,
            threshold=threshold,
            absolute_threshold=absolute_threshold,
            keep_negative=keep_negative,
            self_loops=self_loops
        )
    else:
        G = G.copy()
    
    # Create figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize layout_kws
    if layout_kws is None:
        layout_kws = {}
    
    # Compute graph layout
    if isinstance(layout, dict):
        pos = layout
    elif callable(layout):
        pos = layout(G, **layout_kws)
    elif isinstance(layout, str):
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'shell': nx.shell_layout,
            'spectral': nx.spectral_layout,
            'random': nx.random_layout,
        }
        if layout not in layout_functions:
            raise ValueError(f"Unknown layout: {layout}. Choose from {list(layout_functions.keys())}")
        pos = layout_functions[layout](G, **layout_kws)
    else:
        raise TypeError("layout must be str, dict, or callable")
    
    # Prepare node attributes
    nodes = list(G.nodes())
    
    # Node sizes
    if node_size_col and node_annotations is not None:
        node_sizes = []
        for node in nodes:
            if node in node_annotations.index:
                value = node_annotations.loc[node, node_size_col]
                if node_size_map and value in node_size_map:
                    node_sizes.append(node_size_map[value] * node_size_scale)
                elif pd.notna(value):
                    node_sizes.append(float(value) * node_size_scale)
                else:
                    node_sizes.append(default_node_size)
            else:
                node_sizes.append(default_node_size)
    else:
        node_sizes = [default_node_size] * len(nodes)
    
    # Node colors
    node_colors = []
    node_is_categorical = False
    node_categories = {}
    
    if node_color_col and node_annotations is not None:
        # Check if categorical or continuous
        sample_values = node_annotations[node_color_col].dropna()
        if node_palette or not pd.api.types.is_numeric_dtype(sample_values):
            # Categorical
            node_is_categorical = True
            for node in nodes:
                if node in node_annotations.index:
                    value = node_annotations.loc[node, node_color_col]
                    if pd.notna(value):
                        if value not in node_categories:
                            node_categories[value] = []
                        node_categories[value].append(node)
                        if node_palette and value in node_palette:
                            node_colors.append(node_palette[value])
                        else:
                            node_colors.append(default_node_color)
                    else:
                        node_colors.append(default_node_color)
                else:
                    node_colors.append(default_node_color)
        else:
            # Continuous
            for node in nodes:
                if node in node_annotations.index:
                    value = node_annotations.loc[node, node_color_col]
                    if pd.notna(value):
                        node_colors.append(float(value))
                    else:
                        node_colors.append(np.nan)
                else:
                    node_colors.append(np.nan)
    else:
        node_colors = [default_node_color] * len(nodes)
    
    # Node shapes
    node_shapes = {}
    if node_shape_col and node_annotations is not None:
        for node in nodes:
            if node in node_annotations.index:
                value = node_annotations.loc[node, node_shape_col]
                if node_shape_map and value in node_shape_map:
                    shape = node_shape_map[value]
                else:
                    shape = default_node_shape
            else:
                shape = default_node_shape
            
            if shape not in node_shapes:
                node_shapes[shape] = []
            node_shapes[shape].append(node)
    else:
        node_shapes[default_node_shape] = nodes
    
    # Draw nodes by shape
    for shape, shape_nodes in node_shapes.items():
        shape_indices = [nodes.index(n) for n in shape_nodes]
        shape_pos = {n: pos[n] for n in shape_nodes}
        shape_sizes = [node_sizes[i] for i in shape_indices]
        shape_colors = [node_colors[i] for i in shape_indices]
        
        if not node_is_categorical and node_color_col:
            # Continuous colors - need normalization
            vmin_use = node_vmin if node_vmin is not None else min([c for c in shape_colors if not np.isnan(c)], default=0)
            vmax_use = node_vmax if node_vmax is not None else max([c for c in shape_colors if not np.isnan(c)], default=1)
            
            # Convert node_cmap string to Colormap object
            if isinstance(node_cmap, str):
                from matplotlib import cm
                node_cmap_obj = cm.get_cmap(node_cmap)
            else:
                node_cmap_obj = node_cmap
            
            nx.draw_networkx_nodes(
                G, shape_pos,
                nodelist=shape_nodes,
                node_size=shape_sizes,
                node_color=shape_colors,
                node_shape=shape,
                cmap=node_cmap_obj,  # Use the Colormap object
                vmin=vmin_use,
                vmax=vmax_use,
                alpha=node_alpha,
                edgecolors=node_edgecolors,
                linewidths=node_linewidths,
                ax=ax
            )
        else:
            nx.draw_networkx_nodes(
                G, shape_pos,
                nodelist=shape_nodes,
                node_size=shape_sizes,
                node_color=shape_colors,
                node_shape=shape,
                alpha=node_alpha,
                edgecolors=node_edgecolors,
                linewidths=node_linewidths,
                ax=ax
            )
    
    # Prepare edge attributes
    edges = list(G.edges())
    
    # Edge widths
    if edge_width_col and edge_annotations is not None:
        edge_widths = []
        for source, target in edges:
            # Try both directions for undirected graphs
            mask = ((edge_annotations['source'] == source) & (edge_annotations['target'] == target))
            if not isinstance(G, nx.DiGraph):
                mask = mask | ((edge_annotations['source'] == target) & (edge_annotations['target'] == source))
            
            if mask.any():
                value = edge_annotations.loc[mask, edge_width_col].iloc[0]
                if pd.notna(value):
                    edge_widths.append(float(value) * edge_width_scale)
                else:
                    edge_widths.append(default_edge_width)
            else:
                # Try getting from graph edge attributes
                edge_data = G.get_edge_data(source, target)
                if edge_data and edge_width_col in edge_data:
                    edge_widths.append(float(edge_data[edge_width_col]) * edge_width_scale)
                else:
                    edge_widths.append(default_edge_width)
    elif edge_width_col == 'weight':
        # Use edge weights from graph
        edge_widths = []
        for source, target in edges:
            edge_data = G.get_edge_data(source, target)
            if edge_data and 'weight' in edge_data:
                edge_widths.append(abs(float(edge_data['weight'])) * edge_width_scale)
            else:
                edge_widths.append(default_edge_width)
    else:
        edge_widths = [default_edge_width] * len(edges)
    
    # Edge colors
    edge_colors = []
    edge_is_categorical = False
    edge_categories = {}
    
    if edge_color_col and edge_annotations is not None:
        # Check if categorical or continuous
        sample_values = edge_annotations[edge_color_col].dropna()
        if edge_palette or not pd.api.types.is_numeric_dtype(sample_values):
            # Categorical
            edge_is_categorical = True
            for source, target in edges:
                mask = ((edge_annotations['source'] == source) & (edge_annotations['target'] == target))
                if not isinstance(G, nx.DiGraph):
                    mask = mask | ((edge_annotations['source'] == target) & (edge_annotations['target'] == source))
                
                if mask.any():
                    value = edge_annotations.loc[mask, edge_color_col].iloc[0]
                    if pd.notna(value):
                        if value not in edge_categories:
                            edge_categories[value] = []
                        edge_categories[value].append((source, target))
                        if edge_palette and value in edge_palette:
                            edge_colors.append(edge_palette[value])
                        else:
                            edge_colors.append(default_edge_color)
                    else:
                        edge_colors.append(default_edge_color)
                else:
                    edge_colors.append(default_edge_color)
        else:
            # Continuous
            for source, target in edges:
                mask = ((edge_annotations['source'] == source) & (edge_annotations['target'] == target))
                if not isinstance(G, nx.DiGraph):
                    mask = mask | ((edge_annotations['source'] == target) & (edge_annotations['target'] == source))
                
                if mask.any():
                    value = edge_annotations.loc[mask, edge_color_col].iloc[0]
                    if pd.notna(value):
                        edge_colors.append(float(value))
                    else:
                        edge_colors.append(np.nan)
                else:
                    edge_colors.append(np.nan)
    elif edge_color_col == 'weight':
        # Use edge weights from graph
        for source, target in edges:
            edge_data = G.get_edge_data(source, target)
            if edge_data and 'weight' in edge_data:
                edge_colors.append(float(edge_data['weight']))
            else:
                edge_colors.append(0.0)
    else:
        edge_colors = [default_edge_color] * len(edges)
    
    # Draw edges
    if not edge_is_categorical and edge_color_col:
        # Continuous colors
        vmin_use = edge_vmin if edge_vmin is not None else min([c for c in edge_colors if not np.isnan(c)], default=0)
        vmax_use = edge_vmax if edge_vmax is not None else max([c for c in edge_colors if not np.isnan(c)], default=1)
        
        # Convert edge_cmap string to Colormap object
        if isinstance(edge_cmap, str):
            from matplotlib import cm
            edge_cmap_obj = cm.get_cmap(edge_cmap)
        else:
            edge_cmap_obj = edge_cmap
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=edge_cmap_obj,  # Use the Colormap object
            edge_vmin=vmin_use,
            edge_vmax=vmax_use,
            alpha=edge_alpha,
            style=edge_style,
            arrows=arrows if isinstance(G, nx.DiGraph) else False,
            arrowsize=arrowsize,
            arrowstyle=arrowstyle,
            connectionstyle=connectionstyle,
            ax=ax
        )
    else:
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=edge_alpha,
            style=edge_style,
            arrows=arrows if isinstance(G, nx.DiGraph) else False,
            arrowsize=arrowsize,
            arrowstyle=arrowstyle,
            connectionstyle=connectionstyle,
            ax=ax
        )
    
    # Draw node labels
    if show_node_labels:
        if node_labels is None:
            node_labels = {n: str(n) for n in nodes}
        
        if node_label_position == 'center':
            nx.draw_networkx_labels(
                G, pos,
                labels=node_labels,
                font_size=node_label_size,
                font_color=node_label_color,
                ax=ax
            )
        else:
            # Offset labels above or below nodes
            offset_pos = {}
            for node, (x, y) in pos.items():
                if node_label_position == 'above':
                    offset_pos[node] = (x, y + node_label_offset)
                else:  # below
                    offset_pos[node] = (x, y - node_label_offset)
            
            nx.draw_networkx_labels(
                G, offset_pos,
                labels=node_labels,
                font_size=node_label_size,
                font_color=node_label_color,
                ax=ax
            )
    
    # Draw edge labels
    if show_edge_labels:
        if edge_labels is None and edge_label_col and edge_annotations is not None:
            edge_labels = {}
            for source, target in edges:
                mask = ((edge_annotations['source'] == source) & (edge_annotations['target'] == target))
                if not isinstance(G, nx.DiGraph):
                    mask = mask | ((edge_annotations['source'] == target) & (edge_annotations['target'] == source))
                
                if mask.any():
                    value = edge_annotations.loc[mask, edge_label_col].iloc[0]
                    edge_labels[(source, target)] = str(value)
        elif edge_labels is None:
            edge_labels = {(s, t): f"{G.get_edge_data(s, t).get('weight', ''):.2f}" for s, t in edges if 'weight' in G.get_edge_data(s, t)}
        
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=edge_label_size,
                font_color=edge_label_color,
                ax=ax
            )
    
    # Add legends
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_handles = []
    
    if show_node_legend and node_is_categorical and node_categories:
        for category, cat_nodes in node_categories.items():
            color = node_palette.get(category, default_node_color) if node_palette else default_node_color
            handle = Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=10,
                          label=str(category), linestyle='None')
            legend_handles.append(handle)
    
    if legend_handles:
        ax.legend(handles=legend_handles, title=node_legend_title,
                 loc='center left' if legend_position == 'right' else 'center right',
                 bbox_to_anchor=legend_bbox, fontsize=legend_fontsize)
    
    # Add colorbars
    if show_node_colorbar and not node_is_categorical and node_color_col:
        vmin_use = node_vmin if node_vmin is not None else min([c for c in node_colors if not np.isnan(c)], default=0)
        vmax_use = node_vmax if node_vmax is not None else max([c for c in node_colors if not np.isnan(c)], default=1)
        
        # Convert to Colormap object if string
        if isinstance(node_cmap, str):
            from matplotlib import cm
            node_cmap_obj = cm.get_cmap(node_cmap)
        else:
            node_cmap_obj = node_cmap
        
        sm = ScalarMappable(cmap=node_cmap_obj, norm=Normalize(vmin=vmin_use, vmax=vmax_use))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation=colorbar_orientation, pad=0.05)
        if node_colorbar_label:
            cbar.set_label(node_colorbar_label)
    
    if show_edge_colorbar and not edge_is_categorical and edge_color_col:
        vmin_use = edge_vmin if edge_vmin is not None else min([c for c in edge_colors if not np.isnan(c)], default=0)
        vmax_use = edge_vmax if edge_vmax is not None else max([c for c in edge_colors if not np.isnan(c)], default=1)
        
        # Convert to Colormap object if string
        if isinstance(edge_cmap, str):
            from matplotlib import cm
            edge_cmap_obj = cm.get_cmap(edge_cmap)
        else:
            edge_cmap_obj = edge_cmap
        
        sm = ScalarMappable(cmap=edge_cmap_obj, norm=Normalize(vmin=vmin_use, vmax=vmax_use))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation=colorbar_orientation, pad=0.1)
        if edge_colorbar_label:
            cbar.set_label(edge_colorbar_label)
    
    # Styling
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Prepare info dictionary
    info = {
        'pos': pos,
        'node_categories': node_categories if node_is_categorical else {},
        'edge_categories': edge_categories if edge_is_categorical else {},
        'node_sizes': dict(zip(nodes, node_sizes)),
        'node_colors': dict(zip(nodes, node_colors)),
        'edge_widths': dict(zip(edges, edge_widths)),
        'edge_colors': dict(zip(edges, edge_colors)),
    }
    
    return fig, ax, G, info