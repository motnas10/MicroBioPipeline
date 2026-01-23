"""
Multiplex Network Community Optimizer Library

A flexible framework for optimizing layer weights in multiplex networks to maximize
community detection quality with respect to node attributes.  

Supports edge lists, adjacency matrices, and correlation matrices as input. 

Author: GitHub Copilot
License: MIT
"""

import warnings
from typing import List, Dict, Callable, Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    warnings.warn("igraph not available.  Some community detection methods will not work.")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the unit simplex.
    
    Parameters
    ----------
    v :  np.ndarray
        Input vector to project.
    
    Returns
    -------
    np.ndarray
        Closest point on the simplex to v.
    """
    print('[INFO] Projecting to simplex...')
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def correlation_to_network(
    corr_matrix: np.ndarray,
    node_labels: Optional[List[Any]] = None,
    threshold: Optional[float] = None,
    threshold_percentile: Optional[float] = None,
    keep_top_k: Optional[int] = None,
    absolute_values: bool = False,
    min_weight: Optional[float] = None,
) -> nx.Graph:
    """
    Convert correlation matrix to NetworkX graph.
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix (symmetric, values in [-1, 1]).
    node_labels : Optional[List[Any]], default=None
        Node labels.  If None, uses indices.
    threshold : Optional[float], default=None
        Keep only edges with |correlation| > threshold.
    threshold_percentile : Optional[float], default=None
        Keep only edges above this percentile (0-100).
    keep_top_k : Optional[int], default=None
        Keep only top-k strongest edges per node.
    absolute_values :  bool, default=False
        If True, use absolute correlation values (ignore sign).
    min_weight : Optional[float], default=None
        Minimum absolute weight to include edge (after thresholding).
    
    Returns
    -------
    nx.Graph
        Network with correlation values as edge weights.
    
    Notes
    -----
    Only one of threshold, threshold_percentile, or keep_top_k should be specified. 
    If none specified, includes all non-zero correlations.
    
    Examples
    --------
    >>> corr = np.array([[1.0, 0.8, 0.3],
    ...                  [0.8, 1.0, 0.5],
    ...                  [0.3, 0.5, 1.0]])
    >>> G = correlation_to_network(corr, threshold=0.4)
    """
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    n = corr_matrix.shape[0]
    
    if corr_matrix.shape != (n, n):
        raise ValueError(f"Correlation matrix must be square, got shape {corr_matrix.shape}")
    
    # Create node labels
    if node_labels is None: 
        node_labels = list(range(n))
    elif len(node_labels) != n:
        raise ValueError(f"node_labels length ({len(node_labels)}) doesn't match matrix size ({n})")
    
    # Get upper triangle (excluding diagonal)
    i_upper, j_upper = np.triu_indices(n, k=1)
    correlations = corr_matrix[i_upper, j_upper]
    
    # Apply absolute values if requested
    if absolute_values:
        correlations = np.abs(correlations)
    
    # Apply thresholding
    if threshold is not None: 
        if absolute_values:
            mask = correlations >= threshold
        else:
            mask = np.abs(correlations) >= threshold
        i_upper, j_upper, correlations = i_upper[mask], j_upper[mask], correlations[mask]
    
    elif threshold_percentile is not None: 
        if not 0 <= threshold_percentile <= 100:
            raise ValueError("threshold_percentile must be between 0 and 100")
        threshold_value = np.percentile(np.abs(correlations), threshold_percentile)
        mask = np.abs(correlations) >= threshold_value
        i_upper, j_upper, correlations = i_upper[mask], j_upper[mask], correlations[mask]
    
    elif keep_top_k is not None:
        # Keep top-k edges per node
        edges_per_node = {i: [] for i in range(n)}
        for idx in range(len(i_upper)):
            i, j, w = i_upper[idx], j_upper[idx], correlations[idx]
            edges_per_node[i].append((j, w))
            edges_per_node[j]. append((i, w))
        
        # Sort by absolute correlation and keep top-k
        selected_edges = set()
        for node, edges in edges_per_node. items():
            edges_sorted = sorted(edges, key=lambda x: abs(x[1]), reverse=True)
            for neighbor, weight in edges_sorted[:keep_top_k]:
                edge = tuple(sorted([node, neighbor]))
                selected_edges.add(edge)
        
        # Rebuild edge lists
        i_upper, j_upper, correlations = [], [], []
        for i, j in selected_edges:
            i_upper.append(i)
            j_upper.append(j)
            correlations.append(corr_matrix[i, j])
        
        i_upper = np.array(i_upper)
        j_upper = np.array(j_upper)
        correlations = np. array(correlations)
    
    # Apply minimum weight filter
    if min_weight is not None:
        mask = np. abs(correlations) >= min_weight
        i_upper, j_upper, correlations = i_upper[mask], j_upper[mask], correlations[mask]
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(node_labels)
    
    for idx in range(len(i_upper)):
        u = node_labels[i_upper[idx]]
        v = node_labels[j_upper[idx]]
        w = float(correlations[idx])
        if w != 0:  # Skip zero weights
            G. add_edge(u, v, weight=w)
    
    return G


def signed_sbm_blockwise_sign(
    sizes: List[int],
    p_block: np.ndarray,
    p_pos_block: float = 0.5,
    p_neg_block: float = 0.5,
    w_pos:  float = 1.0,
    w_neg: float = -1.0,
    within_positive:  bool = True,
    return_sign_matrix: bool = False,
    seed: Optional[int] = None,
) -> Union[nx.Graph, Tuple[nx.Graph, np. ndarray]]:
    """
    Generate a signed stochastic block model network.
    
    Parameters
    ----------
    sizes : List[int]
        Number of nodes in each block/community.
    p_block : np.ndarray
        Connection probability matrix between blocks.
    p_pos_block : float, default=0.5
        Probability that a block-pair receives a positive sign.
    p_neg_block : float, default=0.5
        Probability that a block-pair receives a negative sign.
    w_pos : float, default=1.0
        Weight for positive edges.
    w_neg : float, default=-1.0
        Weight for negative edges.
    within_positive : bool, default=True
        If True, within-community edges are always positive.
    return_sign_matrix : bool, default=False
        If True, return both graph and sign matrix.
    seed :  Optional[int], default=None
        Random seed for reproducibility. 
    
    Returns
    -------
    G : nx.Graph or (nx.Graph, np.ndarray)
        Generated signed network, optionally with sign matrix.
    """
    rng = np.random.default_rng(seed)
    
    p_block = np.asarray(p_block, dtype=float)
    K = len(sizes)
    if p_block.shape != (K, K):
        raise ValueError(f"p_block must be shape ({K},{K}), got {p_block.shape}")
    
    if p_pos_block < 0 or p_neg_block < 0 or (p_pos_block + p_neg_block) <= 0:
        raise ValueError("Require p_pos_block >= 0, p_neg_block >= 0, and sum > 0.")
    
    p_pos_n = p_pos_block / (p_pos_block + p_neg_block)
    
    G = nx.stochastic_block_model(sizes, p_block. tolist(), directed=False, selfloops=False, seed=seed)
    
    block_of = nx.get_node_attributes(G, "block")
    if not block_of:
        block_of = {}
        idx = 0
        for b, sz in enumerate(sizes):
            for v in range(idx, idx + sz):
                block_of[v] = b
            idx += sz
    
    S = np.zeros((K, K), dtype=int)
    for a in range(K):
        for b in range(a, K):
            if a == b and within_positive:
                s = +1
            else:
                s = +1 if (rng.random() < p_pos_n) else -1
            S[a, b] = s
            S[b, a] = s
    
    for u, v in G.edges():
        bu, bv = block_of[u], block_of[v]
        s = int(S[bu, bv])
        w = float(w_pos) if s > 0 else float(w_neg)
        G[u][v]["sign"] = s
        G[u][v]["weight"] = w
    
    if return_sign_matrix:
        return G, S
    return G


# =============================================================================
# MAIN CLASS
# =============================================================================

class MultiplexCommunityOptimizer:
    """
    A flexible framework for optimizing multiplex network layer weights. 
    
    This class provides methods to:
    - Load multiplex networks from various sources (Excel, CSV, NetworkX graphs, correlation matrices)
    - Detect communities using various algorithms
    - Evaluate community quality against node attributes
    - Optimize layer weights to maximize community-attribute alignment
    
    Attributes
    ----------
    layers : List[nx.Graph]
        List of network layers (all with same node set).
    layer_names : List[str]
        Names for each layer.
    node_scores : Dict[Any, float]
        Node attribute scores for evaluation.
    weights : np.ndarray
        Current layer weights (sums to 1).
    
    Examples
    --------
    >>> # From Excel file with edge lists
    >>> optimizer = MultiplexCommunityOptimizer. from_excel(
    ...     'networks.xlsx',
    ...     sheets=['layer1', 'layer2'],
    ...     score_column='influence'
    ... )
    >>> 
    >>> # From Excel file with correlation matrices
    >>> optimizer = MultiplexCommunityOptimizer.from_excel(
    ...     'correlations.xlsx',
    ...     sheets=['gene_expr', 'protein'],
    ...     data_format='correlation',
    ...     threshold=0.5
    ... )
    >>> 
    >>> # Optimize weights
    >>> best_weights, best_score = optimizer.optimize(iterations=200)
    >>> print(f"Optimized weights: {best_weights}")
    """
    
    def __init__(
        self,
        layers:  List[nx.Graph],
        node_scores: Optional[Dict[Any, float]] = None,
        layer_names: Optional[List[str]] = None,
        initial_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize the MultiplexCommunityOptimizer.
        
        Parameters
        ----------
        layers : List[nx.Graph]
            List of network layers.  All must have the same node set.
        node_scores : Optional[Dict[Any, float]], default=None
            Mapping from node ID to numeric score. 
        layer_names : Optional[List[str]], default=None
            Names for each layer.  If None, uses "Layer 0", "Layer 1", etc. 
        initial_weights : Optional[np.ndarray], default=None
            Initial layer weights.  If None, uses equal weights.
        
        Raises
        ------
        ValueError
            If layers don't have the same node set.
        """
        if len(layers) == 0:
            raise ValueError("Must provide at least one layer.")
        
        # Validate that all layers have same node set
        node_set = set(layers[0].nodes())
        for i, layer in enumerate(layers[1:], 1):
            if set(layer.nodes()) != node_set:
                raise ValueError(f"Layer {i} has different node set than Layer 0.")
        
        self.layers = layers
        self.node_scores = node_scores or {}
        self.layer_names = layer_names or [f"Layer {i}" for i in range(len(layers))]
        
        # Initialize weights
        if initial_weights is None:
            self.weights = np.ones(len(layers)) / len(layers)
        else:
            self.weights = project_to_simplex(np.asarray(initial_weights))
        
        # Customizable functions
        self.community_detection_func:  Optional[Callable] = None
        self.scoring_func: Optional[Callable] = None
        self.collapse_func: Optional[Callable] = None
        
        # Optimization history
        self.history:  List[Dict[str, Any]] = []
    
    # =========================================================================
    # CLASS METHODS FOR CONSTRUCTION
    # =========================================================================
    
    @classmethod
    def from_excel(
        cls,
        filepath: Union[str, Path],
        sheets: Optional[List[str]] = None,
        data_format: str = 'edgelist',
        score_sheet: Optional[str] = None,
        score_column: Optional[str] = None,
        node_column: str = 'node',
        source_column: str = 'source',
        target_column: str = 'target',
        weight_column: Optional[str] = 'weight',
        # Correlation matrix specific parameters
        threshold: Optional[float] = None,
        threshold_percentile: Optional[float] = None,
        keep_top_k: Optional[int] = None,
        absolute_values: bool = False,
        min_weight: Optional[float] = None,
        **kwargs
    ) -> 'MultiplexCommunityOptimizer': 
        """
        Load multiplex network from Excel file.
        
        Supports two data formats:
        1. Edge list:  Each sheet has source, target, weight columns
        2. Correlation matrix: Each sheet is a correlation matrix
        
        Parameters
        ----------
        filepath : str or Path
            Path to Excel file. 
        sheets : Optional[List[str]], default=None
            List of sheet names to load as layers.  If None, loads all sheets
            except score_sheet. 
        data_format : str, default='edgelist'
            Data format: 'edgelist' or 'correlation'.
        score_sheet : Optional[str], default=None
            Sheet name containing node scores.  If None, no scores loaded.
        score_column : Optional[str], default=None
            Column name for node scores in score_sheet.
        node_column : str, default='node'
            Column name for node IDs in score_sheet.
        source_column : str, default='source'
            Column name for source nodes in edge lists (edgelist format only).
        target_column : str, default='target'
            Column name for target nodes in edge lists (edgelist format only).
        weight_column : Optional[str], default='weight'
            Column name for edge weights (edgelist format only).
        threshold : Optional[float], default=None
            Correlation threshold (correlation format only).
        threshold_percentile : Optional[float], default=None
            Percentile threshold for correlations (correlation format only).
        keep_top_k : Optional[int], default=None
            Keep top-k edges per node (correlation format only).
        absolute_values : bool, default=False
            Use absolute correlation values (correlation format only).
        min_weight : Optional[float], default=None
            Minimum edge weight to include (correlation format only).
        **kwargs
            Additional arguments passed to MultiplexCommunityOptimizer.__init__().
        
        Returns
        -------
        MultiplexCommunityOptimizer
            Initialized optimizer with loaded data.
        
        Examples
        --------
        >>> # Edge list format
        >>> optimizer = MultiplexCommunityOptimizer.from_excel(
        ...      'networks.xlsx',
        ...     sheets=['collaboration', 'communication'],
        ...     data_format='edgelist',
        ...      score_sheet='scores',
        ...     score_column='influence'
        ... )
        >>> 
        >>> # Correlation matrix format
        >>> optimizer = MultiplexCommunityOptimizer. from_excel(
        ...     'correlations.xlsx',
        ...     sheets=['expression', 'methylation'],
        ...     data_format='correlation',
        ...     threshold=0.5,
        ...     absolute_values=True
        ... )
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load Excel file
        excel_file = pd.ExcelFile(filepath)
        all_sheets = excel_file.sheet_names
        
        # Determine which sheets to load as layers
        if sheets is None: 
            sheets = [s for s in all_sheets if s != score_sheet]
        else:
            # Validate sheet names
            missing = set(sheets) - set(all_sheets)
            if missing: 
                raise ValueError(f"Sheets not found in Excel file: {missing}")
        
        # Load layers based on format
        if data_format == 'edgelist':
            layers = cls._load_edgelist_layers(
                filepath, sheets, source_column, target_column, weight_column
            )
        elif data_format == 'correlation': 
            layers = cls._load_correlation_layers(
                filepath, sheets, threshold, threshold_percentile, 
                keep_top_k, absolute_values, min_weight
            )
        else:
            raise ValueError(f"Unknown data_format: {data_format}. Use 'edgelist' or 'correlation'.")
        
        # Load node scores if specified
        node_scores = None
        if score_sheet and score_column:
            if score_sheet not in all_sheets:
                raise ValueError(f"Score sheet '{score_sheet}' not found in Excel file.")
            
            df_scores = pd.read_excel(filepath, sheet_name=score_sheet, header=0)
            if node_column not in df_scores.columns or score_column not in df_scores.columns:
                raise ValueError(
                    f"Score sheet must contain '{node_column}' and '{score_column}' columns."
                )
            
            node_scores = dict(zip(df_scores[node_column], df_scores[score_column]))
        
        return cls(layers=layers, node_scores=node_scores, layer_names=sheets, **kwargs)
    
    @staticmethod
    def _load_edgelist_layers(
        filepath: Path,
        sheets: List[str],
        source_column: str,
        target_column: str,
        weight_column: Optional[str]
    ) -> List[nx.Graph]:
        """Load layers from edge list format."""
        layers = []
        for sheet in sheets:
            df = pd.read_excel(filepath, sheet_name=sheet)
            
            # Validate required columns
            if source_column not in df.columns or target_column not in df. columns:
                raise ValueError(
                    f"Sheet '{sheet}' must contain '{source_column}' and '{target_column}' columns."
                )
            
            # Create graph
            G = nx.Graph()
            for _, row in df.iterrows():
                u, v = row[source_column], row[target_column]
                w = row. get(weight_column, 1.0) if weight_column in df.columns else 1.0
                G.add_edge(u, v, weight=float(w))
            
            layers.append(G)
        
        return layers
    
    @staticmethod
    def _load_correlation_layers(
        filepath: Path,
        sheets: List[str],
        threshold: Optional[float],
        threshold_percentile: Optional[float],
        keep_top_k: Optional[int],
        absolute_values: bool,
        min_weight: Optional[float]
    ) -> List[nx.Graph]:
        """Load layers from correlation matrix format."""
        layers = []
        node_labels = None
        
        for sheet in sheets:
            df = pd.read_excel(filepath, sheet_name=sheet, index_col=0, header=0)
            
            # Validate it's a square matrix
            if df.shape[0] != df.shape[1]:
                raise ValueError(f"Sheet '{sheet}' must be a square correlation matrix.")
            
            # Get node labels from first sheet
            if node_labels is None:
                node_labels = list(df.index)
            else:
                # Validate all sheets have same nodes
                if list(df.index) != node_labels:
                    raise ValueError(f"Sheet '{sheet}' has different node labels than previous sheets.")
            
            # Convert to numpy array
            corr_matrix = df.values
            
            # Convert to network
            G = correlation_to_network(
                corr_matrix,
                node_labels=node_labels,
                threshold=threshold,
                threshold_percentile=threshold_percentile,
                keep_top_k=keep_top_k,
                absolute_values=absolute_values,
                min_weight=min_weight
            )
            
            layers.append(G)
        
        return layers
    
    @classmethod
    def from_csv_folder(
        cls,
        folder_path: Union[str, Path],
        file_pattern: str = "*.csv",
        data_format: str = 'edgelist',
        score_file: Optional[str] = None,
        score_column: Optional[str] = None,
        node_column: str = 'node',
        source_column: str = 'source',
        target_column: str = 'target',
        weight_column: Optional[str] = 'weight',
        # Correlation matrix specific parameters
        threshold: Optional[float] = None,
        threshold_percentile: Optional[float] = None,
        keep_top_k: Optional[int] = None,
        absolute_values: bool = False,
        min_weight: Optional[float] = None,
        **kwargs
    ) -> 'MultiplexCommunityOptimizer': 
        """
        Load multiplex network from folder of CSV files.
        
        Each CSV file represents a network layer in either edge list or
        correlation matrix format.
        
        Parameters
        ----------
        folder_path : str or Path
            Path to folder containing CSV files. 
        file_pattern : str, default="*.csv"
            Glob pattern for selecting CSV files.
        data_format : str, default='edgelist'
            Data format: 'edgelist' or 'correlation'.
        score_file : Optional[str], default=None
            Filename for node scores CSV.  If None, no scores loaded. 
        score_column : Optional[str], default=None
            Column name for scores in score_file.
        node_column : str, default='node'
            Column name for node IDs in score_file.
        source_column : str, default='source'
            Column name for source nodes (edgelist format only).
        target_column : str, default='target'
            Column name for target nodes (edgelist format only).
        weight_column : Optional[str], default='weight'
            Column name for edge weights (edgelist format only).
        threshold :  Optional[float], default=None
            Correlation threshold (correlation format only).
        threshold_percentile :  Optional[float], default=None
            Percentile threshold (correlation format only).
        keep_top_k : Optional[int], default=None
            Keep top-k edges per node (correlation format only).
        absolute_values : bool, default=False
            Use absolute values (correlation format only).
        min_weight : Optional[float], default=None
            Minimum edge weight (correlation format only).
        **kwargs
            Additional arguments passed to __init__().
        
        Returns
        -------
        MultiplexCommunityOptimizer
            Initialized optimizer. 
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find CSV files
        csv_files = list(folder_path.glob(file_pattern))
        if score_file:
            csv_files = [f for f in csv_files if f.name != score_file]
        
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found matching pattern '{file_pattern}'.")
        
        # Load layers based on format
        layers = []
        layer_names = []
        node_labels = None
        
        for csv_file in sorted(csv_files):
            if data_format == 'edgelist':
                df = pd.read_csv(csv_file)
                
                if source_column not in df.columns or target_column not in df. columns:
                    raise ValueError(
                        f"File '{csv_file. name}' must contain '{source_column}' and '{target_column}' columns."
                    )
                
                G = nx.Graph()
                for _, row in df.iterrows():
                    u, v = row[source_column], row[target_column]
                    w = row.get(weight_column, 1.0) if weight_column in df.columns else 1.0
                    G.add_edge(u, v, weight=float(w))
                
                layers.append(G)
            
            elif data_format == 'correlation':
                df = pd. read_csv(csv_file, index_col=0)
                
                if df.shape[0] != df.shape[1]: 
                    raise ValueError(f"File '{csv_file.name}' must be a square correlation matrix.")
                
                if node_labels is None:
                    node_labels = list(df.index)
                else:
                    if list(df.index) != node_labels:
                        raise ValueError(f"File '{csv_file.name}' has different node labels.")
                
                corr_matrix = df.values
                
                G = correlation_to_network(
                    corr_matrix,
                    node_labels=node_labels,
                    threshold=threshold,
                    threshold_percentile=threshold_percentile,
                    keep_top_k=keep_top_k,
                    absolute_values=absolute_values,
                    min_weight=min_weight
                )
                
                layers.append(G)
            
            else:
                raise ValueError(f"Unknown data_format:  {data_format}")
            
            layer_names.append(csv_file.stem)
        
        # Load scores
        node_scores = None
        if score_file and score_column:
            score_path = folder_path / score_file
            if not score_path.exists():
                raise FileNotFoundError(f"Score file not found: {score_path}")
            
            df_scores = pd.read_csv(score_path)
            if node_column not in df_scores. columns or score_column not in df_scores.columns:
                raise ValueError(
                    f"Score file must contain '{node_column}' and '{score_column}' columns."
                )
            
            node_scores = dict(zip(df_scores[node_column], df_scores[score_column]))
        
        return cls(layers=layers, node_scores=node_scores, layer_names=layer_names, **kwargs)
    
    @classmethod
    def from_correlation_matrices(
        cls,
        matrices: List[np.ndarray],
        node_labels: Optional[List[Any]] = None,
        node_scores: Optional[Dict[Any, float]] = None,
        layer_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        threshold_percentile: Optional[float] = None,
        keep_top_k: Optional[int] = None,
        absolute_values: bool = False,
        min_weight: Optional[float] = None,
        **kwargs
    ) -> 'MultiplexCommunityOptimizer': 
        """
        Create optimizer from list of correlation matrices.
        
        Parameters
        ----------
        matrices :  List[np.ndarray]
            List of correlation matrices (same shape, symmetric).
        node_labels : Optional[List[Any]], default=None
            Node labels.  If None, uses integers 0, 1, 2, ...
        node_scores : Optional[Dict[Any, float]], default=None
            Node attribute scores. 
        layer_names : Optional[List[str]], default=None
            Layer names.
        threshold :  Optional[float], default=None
            Keep only edges with |correlation| > threshold.
        threshold_percentile : Optional[float], default=None
            Keep only edges above this percentile (0-100).
        keep_top_k : Optional[int], default=None
            Keep only top-k strongest edges per node.
        absolute_values :  bool, default=False
            Use absolute correlation values. 
        min_weight : Optional[float], default=None
            Minimum absolute weight to include edge.
        **kwargs
            Additional arguments passed to __init__().
        
        Returns
        -------
        MultiplexCommunityOptimizer
            Initialized optimizer.
        
        Examples
        --------
        >>> # Create correlation matrices
        >>> corr1 = np.corrcoef(np.random.randn(50, 100))
        >>> corr2 = np.corrcoef(np. random.randn(50, 100))
        >>> 
        >>> # Create optimizer
        >>> optimizer = MultiplexCommunityOptimizer.from_correlation_matrices(
        ...     matrices=[corr1, corr2],
        ...     threshold=0.5,
        ...      layer_names=['RNA-seq', 'Proteomics']
        ... )
        """
        if len(matrices) == 0:
            raise ValueError("Must provide at least one matrix.")
        
        # Validate shapes
        shape = matrices[0].shape
        for i, mat in enumerate(matrices):
            if mat.shape != shape:
                raise ValueError(f"Matrix {i} has different shape than Matrix 0.")
            if mat.shape[0] != mat.shape[1]:
                raise ValueError(f"Matrix {i} is not square.")
        
        # Create node labels
        n = shape[0]
        if node_labels is None:
            node_labels = list(range(n))
        elif len(node_labels) != n:
            raise ValueError(f"node_labels length ({len(node_labels)}) doesn't match matrix size ({n}).")
        
        # Convert to NetworkX graphs
        layers = []
        for mat in matrices:
            G = correlation_to_network(
                mat,
                node_labels=node_labels,
                threshold=threshold,
                threshold_percentile=threshold_percentile,
                keep_top_k=keep_top_k,
                absolute_values=absolute_values,
                min_weight=min_weight
            )
            layers.append(G)
        
        return cls(layers=layers, node_scores=node_scores, layer_names=layer_names, **kwargs)
    
    @classmethod
    def from_adjacency_matrices(
        cls,
        matrices: List[np.ndarray],
        node_labels: Optional[List[Any]] = None,
        node_scores: Optional[Dict[Any, float]] = None,
        layer_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'MultiplexCommunityOptimizer':
        """
        Create optimizer from list of adjacency matrices. 
        
        Parameters
        ----------
        matrices : List[np.ndarray]
            List of adjacency matrices (same shape).
        node_labels : Optional[List[Any]], default=None
            Node labels. If None, uses integers 0, 1, 2, ...
        node_scores : Optional[Dict[Any, float]], default=None
            Node attribute scores.
        layer_names :  Optional[List[str]], default=None
            Layer names.
        **kwargs
            Additional arguments passed to __init__().
        
        Returns
        -------
        MultiplexCommunityOptimizer
            Initialized optimizer.
        """
        if len(matrices) == 0:
            raise ValueError("Must provide at least one matrix.")
        
        # Validate shapes
        shape = matrices[0].shape
        for i, mat in enumerate(matrices):
            if mat. shape != shape:
                raise ValueError(f"Matrix {i} has different shape than Matrix 0.")
        
        # Create node labels
        n = shape[0]
        if node_labels is None: 
            node_labels = list(range(n))
        elif len(node_labels) != n:
            raise ValueError(f"node_labels length ({len(node_labels)}) doesn't match matrix size ({n}).")
        
        # Convert to NetworkX graphs
        layers = []
        for mat in matrices:
            G = nx.Graph()
            G.add_nodes_from(node_labels)
            for i, u in enumerate(node_labels):
                for j, v in enumerate(node_labels):
                    if i < j and mat[i, j] != 0:
                        G. add_edge(u, v, weight=float(mat[i, j]))
            layers.append(G)
        
        return cls(layers=layers, node_scores=node_scores, layer_names=layer_names, **kwargs)
    
    @classmethod
    def from_synthetic_sbm(
        cls,
        n_layers: int = 2,
        sizes: List[int] = [40, 40, 40],
        p_block: Optional[np.ndarray] = None,
        layer_configs: Optional[List[Dict[str, Any]]] = None,
        generate_scores: bool = True,
        seed: Optional[int] = None,
        **kwargs
    ) -> 'MultiplexCommunityOptimizer':
        """
        Create optimizer with synthetic signed SBM networks.
        
        Parameters
        ----------
        n_layers : int, default=2
            Number of layers to generate.
        sizes : List[int], default=[40, 40, 40]
            Community sizes. 
        p_block : Optional[np.ndarray], default=None
            Connection probability matrix.  If None, generates default.
        layer_configs : Optional[List[Dict]], default=None
            Configuration for each layer (p_pos_block, p_neg_block, etc.).
            If None, generates diverse default configurations.
        generate_scores :  bool, default=True
            If True, generates random node scores.
        seed : Optional[int], default=None
            Random seed for reproducibility.
        **kwargs
            Additional arguments passed to __init__().
        
        Returns
        -------
        MultiplexCommunityOptimizer
            Initialized optimizer with synthetic data.
        
        Examples
        --------
        >>> # Generate 3-layer network with custom configurations
        >>> configs = [
        ...     {'p_pos_block': 0.0, 'p_neg_block': 1.0},  # All negative between communities
        ...     {'p_pos_block': 1.0, 'p_neg_block':  0.0},  # All positive between communities
        ...     {'p_pos_block': 0.5, 'p_neg_block': 0.5},  # Mixed
        ... ]
        >>> optimizer = MultiplexCommunityOptimizer.from_synthetic_sbm(
        ...     n_layers=3,
        ...     layer_configs=configs,
        ...      seed=42
        ... )
        """
        rng = np.random.default_rng(seed)
        
        # Default p_block
        if p_block is None:
            K = len(sizes)
            p_block = np.full((K, K), 0.03)
            np.fill_diagonal(p_block, 0.25)
        
        # Default layer configs
        if layer_configs is None:
            layer_configs = []
            for i in range(n_layers):
                # Create diverse configurations
                if i % 3 == 0:
                    config = {'p_pos_block': 0.0, 'p_neg_block': 1.0}  # Negative between
                elif i % 3 == 1:
                    config = {'p_pos_block': 1.0, 'p_neg_block': 0.0}  # Positive between
                else: 
                    config = {'p_pos_block': 0.5, 'p_neg_block':  0.5}  # Mixed
                layer_configs.append(config)
        
        # Generate layers
        layers = []
        for i, config in enumerate(layer_configs[: n_layers]):
            G = signed_sbm_blockwise_sign(
                sizes=sizes,
                p_block=p_block,
                within_positive=True,
                seed=None if seed is None else seed + i,
                **config
            )
            layers.append(G)
        
        # Generate scores
        node_scores = None
        if generate_scores: 
            nodes = list(layers[0].nodes())
            node_scores = {u: float(rng.random()) for u in nodes}
        
        return cls(layers=layers, node_scores=node_scores, **kwargs)
    
    # =========================================================================
    # LAYER MANIPULATION
    # =========================================================================
    
    def collapse_layers(
        self,
        weights: Optional[np.ndarray] = None,
        custom_collapse: Optional[Callable] = None
    ) -> nx.Graph:
        """
        Collapse multiple layers into single weighted network.
        
        Parameters
        ----------
        weights : Optional[np.ndarray], default=None
            Layer weights. If None, uses self.weights.
        custom_collapse : Optional[Callable], default=None
            Custom collapse function with signature: 
            f(layers:  List[nx.Graph], weights: np.ndarray) -> nx.Graph
            If None, uses default weighted sum.
        
        Returns
        -------
        nx.Graph
            Collapsed network.
        """
        print('[INFO] Collapsing layers into single network...')
        if weights is None:
            weights = self.weights
        else:
            weights = project_to_simplex(np.asarray(weights))
        
        # Use custom function if provided
        if custom_collapse is not None:
            return custom_collapse(self.layers, weights)
        
        # Use class-level custom function if set
        if self.collapse_func is not None:
            return self.collapse_func(self.layers, weights)
        
        # Default:  weighted sum of adjacency matrices
        A_list = [nx.to_numpy_array(g, weight="weight", dtype=float) for g in self.layers]
        A_list = [weights[i] * A_list[i] for i in range(len(A_list))]
        A = np.sum(np.array(A_list), axis=0)
        return nx.from_numpy_array(A)
    
    def set_collapse_function(self, func: Callable) -> None:
        """
        Set custom layer collapse function.
        
        Parameters
        ----------
        func :  Callable
            Function with signature:  f(layers, weights) -> nx.Graph
        
        Examples
        --------
        >>> def max_collapse(layers, weights):
        ...     # Take max weight across layers
        ...     matrices = [nx.to_numpy_array(g) for g in layers]
        ...      A = np.maximum.reduce([w * m for w, m in zip(weights, matrices)])
        ...     return nx.from_numpy_array(A)
        >>>
        >>> optimizer.set_collapse_function(max_collapse)
        """
        self.collapse_func = func
    
    # =========================================================================
    # COMMUNITY DETECTION
    # =========================================================================
    
    def detect_communities(
        self,
        G: Optional[nx.Graph] = None,
        method: str = 'spinglass',
        custom_method: Optional[Callable] = None,
        **method_kwargs
    ) -> List[List[Any]]:
        """
        Detect communities in a network.
        
        Parameters
        ----------
        G : Optional[nx.Graph], default=None
            Network to analyze. If None, uses collapsed network.
        method : str, default='spinglass'
            Community detection method: 
            - 'spinglass': Spinglass algorithm (requires igraph)
            - 'louvain': Louvain algorithm
            - 'greedy_modularity': Greedy modularity optimization
            - 'label_propagation': Label propagation
        custom_method : Optional[Callable], default=None
            Custom community detection function with signature:
            f(G: nx.Graph, **kwargs) -> List[List[Any]]
        **method_kwargs
            Additional arguments for the detection method.
        
        Returns
        -------
        List[List[Any]]
            List of communities (each community is a list of nodes).
        
        Examples
        --------
        >>> # Use built-in method
        >>> communities = optimizer.detect_communities(method='louvain')
        >>>
        >>> # Use custom method
        >>> def my_method(G, threshold=0.5):
        ...     # Custom logic
        ...     return [[0, 1, 2], [3, 4, 5]]
        >>>
        >>> communities = optimizer.detect_communities(custom_method=my_method, threshold=0.6)
        """
        print('[INFO] Detecting communities...')
        if G is None:
            G = self.collapse_layers()
        
        # Use custom function if provided
        if custom_method is not None:
            return custom_method(G, **method_kwargs)
        
        # Use class-level custom function if set
        if self.community_detection_func is not None:
            return self.community_detection_func(G, **method_kwargs)
        
        # Built-in methods
        if method == 'spinglass':
            return self._detect_spinglass(G, **method_kwargs)
        elif method == 'louvain': 
            return self._detect_louvain(G, **method_kwargs)
        elif method == 'greedy_modularity': 
            return self._detect_greedy_modularity(G, **method_kwargs)
        elif method == 'label_propagation': 
            return self._detect_label_propagation(G, **method_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_spinglass(self, G:  nx.Graph, **kwargs) -> List[List[Any]]:
        """Spinglass community detection (requires igraph)."""
        print('[INFO] Using spinglass community detection...')
        if not IGRAPH_AVAILABLE:
            raise ImportError("igraph is required for spinglass method.  Install with: pip install igraph")
        
        nodes = list(G.nodes())
        node_index = {v: i for i, v in enumerate(nodes)}
        
        edges = [(node_index[u], node_index[v]) for u, v in G.edges()]
        weights = [float(G[u][v]. get("weight", 1.0)) for u, v in G.edges()]
        
        g = ig.Graph(n=len(nodes), edges=edges, directed=False)
        g.es["weight"] = weights
        
        # Default parameters
        params = {'weights': 'weight', 'implementation': 'neg'}
        params.update(kwargs)
        
        clusters = g.community_spinglass(**params)
        communities = [[nodes[i] for i in c] for c in clusters]
        return communities
    
    def _detect_louvain(self, G: nx.Graph, **kwargs) -> List[List[Any]]: 
        """Louvain community detection."""
        print('[INFO] Using louvain community detection...')
        try:
            import community as community_louvain
        except ImportError: 
            raise ImportError("python-louvain is required.  Install with: pip install python-louvain")
        
        partition = community_louvain.best_partition(G, weight='weight', **kwargs)
        communities_dict = {}
        for node, comm in partition.items():
            communities_dict. setdefault(comm, []).append(node)
        return list(communities_dict.values())
    
    def _detect_greedy_modularity(self, G: nx.Graph, **kwargs) -> List[List[Any]]: 
        """Greedy modularity optimization."""
        print('[INFO] Using greedy modularity community detection...')
        communities_gen = nx.algorithms.community.greedy_modularity_communities(G, weight='weight', **kwargs)
        return [list(c) for c in communities_gen]
    
    def _detect_label_propagation(self, G: nx.Graph, **kwargs) -> List[List[Any]]:
        """Label propagation."""
        print('[INFO] Using label propagation community detection...')
        communities_gen = nx.algorithms.community.label_propagation_communities(G, weight='weight', **kwargs)
        return [list(c) for c in communities_gen]
    
    def set_community_detection_function(self, func: Callable) -> None:
        """
        Set custom community detection function.
        
        Parameters
        ----------
        func :  Callable
            Function with signature:  f(G:  nx.Graph, **kwargs) -> List[List[Any]]
        
        Examples
        --------
        >>> def my_detector(G, resolution=1. 0):
        ...     # Custom community detection
        ...     return [[0, 1], [2, 3, 4]]
        >>>
        >>> optimizer.set_community_detection_function(my_detector)
        """
        self.community_detection_func = func
    
    # =========================================================================
    # SCORING
    # =========================================================================
    
    def score_communities(
        self,
        communities: List[List[Any]],
        node_scores: Optional[Dict[Any, float]] = None,
        custom_scorer: Optional[Callable] = None,
        metric: str = 'r2'
    ) -> float:
        """
        Evaluate quality of community partition against node scores.
        
        Parameters
        ----------
        communities : List[List[Any]]
            Community partition.
        node_scores :  Optional[Dict[Any, float]], default=None
            Node scores.  If None, uses self.node_scores.
        custom_scorer :  Optional[Callable], default=None
            Custom scoring function with signature: 
            f(communities, node_scores) -> float
        metric : str, default='r2'
            Scoring metric:
            - 'r2': R² (fraction of variance explained)
            - 'silhouette': Silhouette score
            - 'modularity': Modularity score (doesn't use node_scores)
        
        Returns
        -------
        float
            Quality score (higher is better).
        
        Examples
        --------
        >>> # Use R² metric
        >>> score = optimizer.score_communities(communities, metric='r2')
        >>>
        >>> # Use custom scorer
        >>> def my_scorer(communities, scores):
        ...     # Custom logic
        ...     return 0.85
        >>>
        >>> score = optimizer.score_communities(communities, custom_scorer=my_scorer)
        """
        print('[INFO] Scoring communities...')
        if node_scores is None:
            node_scores = self.node_scores
        
        if len(node_scores) == 0 and metric != 'modularity':
            raise ValueError("node_scores is empty.  Provide scores or use metric='modularity'.")
        
        # Use custom function if provided
        if custom_scorer is not None:
            return custom_scorer(communities, node_scores)
        
        # Use class-level custom function if set
        if self.scoring_func is not None:
            return self.scoring_func(communities, node_scores)
        
        # Built-in metrics
        if metric == 'r2':
            return self._score_r2(communities, node_scores)
        elif metric == 'silhouette':
            return self._score_silhouette(communities, node_scores)
        elif metric == 'modularity': 
            G = self.collapse_layers()
            return self._score_modularity(G, communities)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _score_r2(self, communities:  List[List[Any]], node_scores: Dict[Any, float]) -> float:
        """R² score (fraction of variance explained)."""
        print('[INFO] Calculating R² score...')
        labels = {}
        for c, nodes in enumerate(communities):
            for u in nodes:
                labels[u] = c
        
        scores = np.array([node_scores[u] for u in labels if u in node_scores])
        comms = np.array([labels[u] for u in labels if u in node_scores])
        
        if len(scores) == 0:
            return 0.0
        
        var_total = scores.var(ddof=0)
        if var_total == 0:
            return 0.0
        
        var_within = np.mean([
            scores[comms == c].var(ddof=0)
            for c in np.unique(comms)
            if np.sum(comms == c) > 0
        ])
        
        R2 = 1.0 - var_within / var_total
        return float(R2)
    
    def _score_silhouette(self, communities: List[List[Any]], node_scores: Dict[Any, float]) -> float:
        """Silhouette score."""
        print('[INFO] Calculating silhouette score...')
        from sklearn.metrics import silhouette_score
        
        labels = {}
        for c, nodes in enumerate(communities):
            for u in nodes:
                labels[u] = c
        
        nodes_list = [u for u in labels if u in node_scores]
        if len(nodes_list) < 2:
            return 0.0
        
        scores = np.array([[node_scores[u]] for u in nodes_list])
        comms = np.array([labels[u] for u in nodes_list])
        
        if len(np.unique(comms)) < 2:
            return 0.0
        
        return float(silhouette_score(scores, comms))
    
    def _score_modularity(self, G: nx.Graph, communities: List[List[Any]]) -> float:
        """Modularity score."""
        print('[INFO] Calculating modularity score...')
        return float(nx.algorithms.community.modularity(G, communities, weight='weight'))
    
    def set_scoring_function(self, func: Callable) -> None:
        """
        Set custom scoring function.
        
        Parameters
        ----------
        func : Callable
            Function with signature: f(communities, node_scores) -> float
        
        Examples
        --------
        >>> def my_scorer(communities, node_scores):
        ...     # Custom scoring logic
        ...     return 0.75
        >>>
        >>> optimizer.set_scoring_function(my_scorer)
        """
        self.scoring_func = func
    
    # =========================================================================
    # OPTIMIZATION
    # =========================================================================
    
    def objective(
        self,
        weights: np.ndarray,
        repeats: int = 1,
        community_method: str = 'spinglass',
        scoring_metric: str = 'r2',
        verbose: bool = False,
        **kwargs
    ) -> float:
        """
        Objective function for optimization.
        
        Parameters
        ----------
        weights : np.ndarray
            Layer weights to evaluate.
        repeats : int, default=1
            Number of evaluations to average (for stochastic methods).
        community_method : str, default='spinglass'
            Community detection method. 
        scoring_metric : str, default='r2'
            Scoring metric.
        verbose : bool, default=False
            Print intermediate results.
        **kwargs
            Additional arguments for detection/scoring. 
        
        Returns
        -------
        float
            Average score across repeats.
        """
        print('[INFO] Evaluating objective function...')
        weights = project_to_simplex(weights)
        
        vals = []
        for i in tqdm(range(repeats), total=repeats, desc="Repeats"):
            G = self.collapse_layers(weights)
            communities = self.detect_communities(G, method=community_method, **kwargs)
            score = self.score_communities(communities, metric=scoring_metric)
            vals.append(float(score))
            
            if verbose:
                print(f"  Repeat {i+1}/{repeats}: score = {score:.4f}")
        
        return float(np.mean(vals))
    
    def optimize(
        self,
        method: str = 'spsa',
        iterations: int = 500,
        repeats: int = 1,
        community_method: str = 'spinglass',
        scoring_metric: str = 'r2',
        verbose: bool = True,
        log_frequency: int = 20,
        seed: Optional[int] = None,
        **optimizer_kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize layer weights to maximize community-score alignment.
        
        Parameters
        ----------
        method : str, default='spsa'
            Optimization method:
            - 'spsa': Simultaneous Perturbation Stochastic Approximation
            - 'grid': Grid search (exhaustive for 2-3 layers)
            - 'random': Random search
        iterations :  int, default=200
            Number of optimization iterations.
        repeats : int, default=1
            Objective function evaluations per iteration.
        community_method :  str, default='spinglass'
            Community detection method.
        scoring_metric : str, default='r2'
            Scoring metric.
        verbose : bool, default=True
            Print progress.
        log_frequency : int, default=20
            Print every N iterations.
        seed : Optional[int], default=None
            Random seed. 
        **optimizer_kwargs
            Additional arguments for optimizer (e.g., alpha, c for SPSA).
        
        Returns
        -------
        best_weights : np.ndarray
            Optimized layer weights.
        best_score : float
            Best objective value achieved.
        
        Examples
        --------
        >>> # Basic optimization
        >>> weights, score = optimizer.optimize(iterations=100)
        >>> print(f"Best weights: {weights}, Score: {score:.4f}")
        >>>
        >>> # With custom parameters
        >>> weights, score = optimizer.optimize(
        ...     method='spsa',
        ...     iterations=500,
        ...     repeats=3,
        ...     alpha=0.3,
        ...     c=0.05,
        ...     seed=42
        ... )
        """
        print('[INFO] Starting optimization...')
        if method == 'spsa':
            return self._optimize_spsa(
                iterations=iterations,
                repeats=repeats,
                community_method=community_method,
                scoring_metric=scoring_metric,
                verbose=verbose,
                log_frequency=log_frequency,
                seed=seed,
                **optimizer_kwargs
            )
        elif method == 'grid': 
            return self._optimize_grid(
                community_method=community_method,
                scoring_metric=scoring_metric,
                verbose=verbose,
                **optimizer_kwargs
            )
        elif method == 'random':
            return self._optimize_random(
                iterations=iterations,
                community_method=community_method,
                scoring_metric=scoring_metric,
                verbose=verbose,
                seed=seed,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_spsa(
        self,
        iterations: int,
        repeats: int,
        community_method: str,
        scoring_metric: str,
        verbose: bool,
        log_frequency: int,
        seed: Optional[int],
        alpha: float = 0.2,
        c: float = 0.1,
        alpha_decay: float = 0.602,
        c_decay: float = 0.101,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """SPSA optimization."""
        print('[INFO] Running SPSA optimization...')
        rng = np.random.default_rng(seed)
        
        m = len(self.layers)
        a = self.weights.copy()
        
        best_a = a.copy()
        best_f = -np.inf
        
        self.history = []
        
        if verbose:
            print("=" * 70)
            print("SPSA OPTIMIZATION")
            print("=" * 70)
            print(f"Layers: {m}, Iterations: {iterations}, Repeats:  {repeats}")
            print(f"Parameters: alpha={alpha}, c={c}, alpha_decay={alpha_decay}, c_decay={c_decay}")
            print(f"Detection:  {community_method}, Metric: {scoring_metric}")
            print()
        
        for t in tqdm(range(1, iterations + 1), total=iterations, desc="Iterations"):
            delta = rng.choice([-1.0, 1.0], size=m)
            
            at = alpha / (t ** alpha_decay)
            ct = c / (t ** c_decay)
            
            print('[INFO] Computing a_plus and a_minus...')
            a_plus = project_to_simplex(a + ct * delta)
            a_minus = project_to_simplex(a - ct * delta)
            
            print('[INFO] Evaluating objective function...')
            f_plus = self.objective(a_plus, repeats=repeats, community_method=community_method, scoring_metric=scoring_metric, **kwargs)
            f_minus = self.objective(a_minus, repeats=repeats, community_method=community_method, scoring_metric=scoring_metric, **kwargs)
            
            ghat = (f_plus - f_minus) / (2.0 * ct) * delta
            
            a = project_to_simplex(a + at * ghat)
            
            f = self.objective(a, repeats=repeats, community_method=community_method, scoring_metric=scoring_metric, **kwargs)
            
            if f > best_f:
                best_f, best_a = f, a. copy()
            
            self.history.append({
                'iteration': t,
                'weights': a.copy(),
                'score': f,
                'best_score': best_f,
            })
            
            if verbose and (t % log_frequency == 0 or t == 1):
                weights_str = ', '.join([f"{w:.3f}" for w in a])
                print(f"  Iter {t: 3d}: score={f:.4f}, best={best_f:.4f}, weights=[{weights_str}]")
        
        self.weights = best_a
        
        if verbose:
            print("\n" + "=" * 70)
            print("OPTIMIZATION COMPLETE")
            print("=" * 70)
            print(f"Best score: {best_f:.4f}")
            print(f"Best weights: {best_a}")
            print()
        
        return best_a, best_f
    
    def _optimize_grid(
        self,
        community_method:  str,
        scoring_metric:  str,
        verbose: bool,
        grid_points: int = 11,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """Grid search optimization (only practical for 2-3 layers)."""
        m = len(self.layers)
        
        if m > 3:
            warnings.warn(f"Grid search with {m} layers will be very slow. Consider using method='spsa' instead.")
        
        if verbose:
            print("=" * 70)
            print("GRID SEARCH OPTIMIZATION")
            print("=" * 70)
            print(f"Layers: {m}, Grid points: {grid_points}")
            print()
        
        # Generate grid
        from itertools import product
        grid_1d = np.linspace(0, 1, grid_points)
        
        best_weights = None
        best_score = -np.inf
        
        total_points = grid_points ** (m - 1)
        evaluated = 0
        
        for weights_raw in product(grid_1d, repeat=m-1):
            weights = np.array(list(weights_raw) + [1.0 - sum(weights_raw)])
            
            if np.any(weights < 0):
                continue
            
            weights = project_to_simplex(weights)
            
            score = self.objective(weights, repeats=1, community_method=community_method, scoring_metric=scoring_metric, **kwargs)
            evaluated += 1
            
            if score > best_score:
                best_score = score
                best_weights = weights. copy()
            
            if verbose and evaluated % max(1, total_points // 10) == 0:
                print(f"  Evaluated {evaluated}/{total_points}:  best_score={best_score:.4f}")
        
        self.weights = best_weights
        
        if verbose:
            print("\n" + "=" * 70)
            print("GRID SEARCH COMPLETE")
            print("=" * 70)
            print(f"Best score: {best_score:.4f}")
            print(f"Best weights:  {best_weights}")
            print()
        
        return best_weights, best_score
    
    def _optimize_random(
        self,
        iterations: int,
        community_method: str,
        scoring_metric: str,
        verbose: bool,
        seed: Optional[int],
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """Random search optimization."""
        rng = np.random.default_rng(seed)
        
        m = len(self.layers)
        
        best_weights = None
        best_score = -np.inf
        
        if verbose:
            print("=" * 70)
            print("RANDOM SEARCH OPTIMIZATION")
            print("=" * 70)
            print(f"Layers: {m}, Iterations: {iterations}")
            print()
        
        for t in range(1, iterations + 1):
            # Sample random weights from Dirichlet distribution
            weights = rng.dirichlet(np. ones(m))
            
            score = self.objective(weights, repeats=1, community_method=community_method, scoring_metric=scoring_metric, **kwargs)
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
            
            if verbose and (t % 20 == 0 or t == 1):
                weights_str = ', '.join([f"{w:.3f}" for w in weights])
                print(f"  Iter {t:3d}: score={score:. 4f}, best={best_score:.4f}, weights=[{weights_str}]")
        
        self.weights = best_weights
        
        if verbose: 
            print("\n" + "=" * 70)
            print("RANDOM SEARCH COMPLETE")
            print("=" * 70)
            print(f"Best score: {best_score:.4f}")
            print(f"Best weights: {best_weights}")
            print()
        
        return best_weights, best_score
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            History with columns:  iteration, score, best_score, and weights.
        """
        if len(self.history) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        
        # Expand weights into separate columns
        for i in range(len(self. layers)):
            df[f'weight_{i}'] = df['weights'].apply(lambda w: w[i])
        
        df = df.drop(columns=['weights'])
        return df
    
    def plot_optimization_history(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot optimization history.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(12, 5)
            Figure size.
        """
        import matplotlib.pyplot as plt
        
        df = self.get_optimization_history()
        if df.empty:
            print("No optimization history to plot.  Run optimize() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot scores
        ax1.plot(df['iteration'], df['score'], label='Current', alpha=0.6)
        ax1.plot(df['iteration'], df['best_score'], label='Best', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.set_title('Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot weights
        weight_cols = [c for c in df.columns if c.startswith('weight_')]
        for col in weight_cols:
            layer_idx = int(col.split('_')[1])
            ax2.plot(df['iteration'], df[col], label=self.layer_names[layer_idx], alpha=0.8)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Weight')
        ax2.set_title('Layer Weights Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> None:
        """Print summary of current state."""
        print("=" * 70)
        print("MULTIPLEX COMMUNITY OPTIMIZER SUMMARY")
        print("=" * 70)
        print(f"\nLayers: {len(self.layers)}")
        
        for i, name in enumerate(self.layer_names):
            G = self.layers[i]
            print(f"  {i}. {name}:  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        print(f"\nNode scores: {len(self.node_scores)} nodes")
        if len(self.node_scores) > 0:
            scores = list(self.node_scores.values())
            print(f"  Range: [{min(scores):.3f}, {max(scores):.3f}]")
            print(f"  Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
        
        print(f"\nCurrent weights:")
        for i, (name, w) in enumerate(zip(self.layer_names, self.weights)):
            print(f"  {name}: {w:.4f} ({w*100:.1f}%)")
        
        print(f"\nOptimization history:  {len(self.history)} iterations")
        
        print("\nCustom functions:")
        print(f"  Collapse:  {'Yes' if self.collapse_func else 'No'}")
        print(f"  Community detection: {'Yes' if self.community_detection_func else 'No'}")
        print(f"  Scoring:  {'Yes' if self.scoring_func else 'No'}")
        
        print("=" * 70)
    
    def export_results(self, filepath: Union[str, Path]) -> None:
        """
        Export optimization results to Excel file.
        
        Parameters
        ----------
        filepath : str or Path
            Output filepath.
        """
        filepath = Path(filepath)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Weights summary
            weights_df = pd.DataFrame({
                'Layer': self.layer_names,
                'Weight':  self.weights,
                'Percentage': self.weights * 100
            })
            weights_df. to_excel(writer, sheet_name='Weights', index=False)
            
            # Optimization history
            if len(self.history) > 0:
                history_df = self.get_optimization_history()
                history_df.to_excel(writer, sheet_name='History', index=False)
            
            # Node scores
            if len(self.node_scores) > 0:
                scores_df = pd.DataFrame({
                    'Node': list(self.node_scores.keys()),
                    'Score': list(self.node_scores.values())
                })
                scores_df.to_excel(writer, sheet_name='Node_Scores', index=False)
        
        print(f"Results exported to:  {filepath}")



