import pandas as pd


# --------------------------------------------------------------------------------------------------------------
# Discretization of numeric columns in DataFrame
def discretize_equal_width(df, columns, n_levels):
    """
    Discretize numeric columns using equal-width bins.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to discretize.
        n_levels (int): Number of bins.
    """
    df_discrete = df.copy()
    for col in columns:
        df_discrete[col] = pd.cut(df[col], bins=n_levels, labels=False) - 1
    return df_discrete


# --------------------------------------------------------------------------------------------------------------
# Remove rows/columns with low sums from square DataFrame
def filter_zero_nodes_df(df, min_sum=1):
    """
    Remove rows/columns whose row-sum is < min_sum.
    Accepts a square pandas DataFrame (rows and columns same labels).
    Returns filtered DataFrame and a list of kept labels.
    """
    row_sum = df.sum(axis=1)
    keep = row_sum >= min_sum
    kept_labels = df.index[keep].tolist()
    filtered = df.loc[kept_labels, kept_labels].copy()
    return filtered, kept_labels


# --------------------------------------------------------------------------------------------------------------
# Regularization
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize_scalar
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class NetworkRegularizer:
    """
    Implementation of regularization methods from Santucci et al. (2020)
    for estimating partial correlations in network analysis.
    """
    
    def __init__(self, method='identity_shrinkage', correlation_axis='columns', 
                 test_size=0.2, random_state=42):
        """
        Initialize the regularizer.
        
        Parameters:
        -----------
        method : str
            Regularization method: 'identity_shrinkage', 'group_shrinkage', 
            'graphical_lasso', 'partial_correlation', or 'ledoit_wolf'
        correlation_axis : str
            'columns' for feature-feature correlation (default)
            'rows' for sample-sample correlation
        test_size : float
            Proportion of data to use for validation
        random_state : int
            Random seed for reproducibility
        """
        self.method = method
        self.correlation_axis = correlation_axis
        self.test_size = test_size
        self.random_state = random_state
        self.alpha_optimal = None
        self.C_mu = None
        self.J_mu = None
        self.J_tilde = None
        self.sample_corr = None
        self.final_matrix = None
        
    def prepare_data(self, X):
        """
        Phase I: Data Preparation and Standardization
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Raw N×T data matrix (N samples × T features)
            
        Returns:
        --------
        X_standardized : numpy array
            Demeaned and standardized data
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Store sample names if working with rows
        self.sample_names = [f'Sample_{i}' for i in range(X.shape[0])]
        
        # Check for and handle NaN/Inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: Data contains NaN or Inf values. Cleaning...")
            
            # Remove columns with all NaN
            col_mask = ~np.all(np.isnan(X), axis=0)
            X = X[:, col_mask]
            self.feature_names = [name for name, keep in zip(self.feature_names, col_mask) if keep]
            print(f"  Removed {(~col_mask).sum()} columns with all NaN values")
            
            # Remove rows with any NaN
            row_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
            X = X[row_mask, :]
            self.sample_names = [name for name, keep in zip(self.sample_names, row_mask) if keep]
            print(f"  Removed {(~row_mask).sum()} rows with NaN/Inf values")
            
            if X.shape[0] < 10:
                raise ValueError("Too few valid samples remaining after cleaning")
            if X.shape[1] < 2:
                raise ValueError("Too few valid features remaining after cleaning")
        
        # Transpose if correlating rows
        if self.correlation_axis == 'rows':
            print(f"Computing correlation across ROWS (sample-sample correlation)")
            X = X.T  # Transpose so rows become columns
            self.feature_names, self.sample_names = self.sample_names, self.feature_names
        else:
            print(f"Computing correlation across COLUMNS (feature-feature correlation)")
        
        # Check for zero variance columns
        col_std = np.std(X, axis=0, ddof=1)
        zero_var_mask = col_std > 1e-10
        
        if not np.all(zero_var_mask):
            print(f"Warning: Removing {(~zero_var_mask).sum()} zero-variance columns")
            X = X[:, zero_var_mask]
            self.feature_names = [name for name, keep in zip(self.feature_names, zero_var_mask) if keep]
        
        # Demean and standardize to zero mean and unit variance
        X_standardized = zscore(X, axis=0, ddof=1)
        
        # Final check
        if np.any(np.isnan(X_standardized)) or np.any(np.isinf(X_standardized)):
            raise ValueError("Standardization produced NaN/Inf values. Check your data quality.")
        
        return X_standardized
    
    def compute_sample_correlation(self, X_std):
        """
        Calculate sample correlation matrix E
        
        Parameters:
        -----------
        X_std : numpy array
            Standardized data matrix
            
        Returns:
        --------
        E : numpy array
            Sample correlation matrix
        """
        N, T = X_std.shape
        E = (X_std.T @ X_std) / N
        
        # Ensure numerical stability
        E = (E + E.T) / 2  # Enforce symmetry
        np.fill_diagonal(E, 1.0)  # Ensure diagonal is exactly 1
        
        # Clip to valid correlation range
        E = np.clip(E, -1, 1)
        
        return E
    
    def identity_shrinkage(self, E, alpha):
        """
        Identity Shrinkage regularization
        
        C_IS(α) = (1-α)E + αI
        """
        N = E.shape[0]
        I = np.eye(N)
        return (1 - alpha) * E + alpha * I
    
    def group_shrinkage(self, E, alpha, n_groups=5):
        """
        Group Shrinkage regularization using hierarchical clustering
        
        Parameters:
        -----------
        E : numpy array
            Sample correlation matrix
        alpha : float
            Shrinkage parameter
        n_groups : int
            Number of groups for clustering
        """
        # Perform hierarchical clustering
        distance_matrix = 1 - np.abs(E)
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed distance matrix for linkage
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_dist, method='average')
        
        # Get cluster assignments
        clusters = fcluster(Z, n_groups, criterion='maxclust')
        
        # Compute group averages
        C_group = E.copy()
        for i in range(1, n_groups + 1):
            mask = clusters == i
            group_indices = np.where(mask)[0]
            
            if len(group_indices) > 1:
                # Average correlations within group
                for idx1 in group_indices:
                    for idx2 in group_indices:
                        if idx1 != idx2:
                            group_mean = E[np.ix_(group_indices, group_indices)].mean()
                            C_group[idx1, idx2] = group_mean
        
        # Shrink towards group structure
        return (1 - alpha) * E + alpha * C_group
    
    def graphical_lasso_reg(self, E, alpha):
        """
        Graphical LASSO regularization (sparse inverse covariance)
        """
        try:
            from sklearn.covariance import graphical_lasso
            # GLASSO works on covariance, E is already correlation
            cov_estimated, precision = graphical_lasso(E, alpha=alpha, max_iter=200)
            
            # Return the estimated covariance/correlation
            return cov_estimated
        except Exception as e:
            print(f"  Warning: GLASSO failed ({str(e)}), using identity shrinkage")
            # Fallback to identity shrinkage if GLASSO fails
            return self.identity_shrinkage(E, alpha)
    
    def ledoit_wolf_reg(self, X_std):
        """
        Ledoit-Wolf regularization (optimal shrinkage)
        """
        cov, _ = ledoit_wolf(X_std)
        # Convert to correlation
        d = np.sqrt(np.diag(cov))
        C = cov / np.outer(d, d)
        return C
    
    def log_likelihood(self, C, E_val, N_val):
        """
        Compute log-likelihood for validation
        
        L = -N/2 * (log|C| + tr(C^{-1}E_val))
        """
        try:
            sign, logdet = np.linalg.slogdet(C)
            if sign <= 0:
                return -np.inf
            
            C_inv = linalg.inv(C)
            trace_term = np.trace(C_inv @ E_val)
            
            log_lik = -N_val / 2 * (logdet + trace_term)
            return log_lik
        except:
            return -np.inf
    
    def find_optimal_alpha(self, X_train, X_val, alpha_range=(0.001, 0.99)):
        """
        Phase II: Find optimal regularization parameter α_L
        
        Parameters:
        -----------
        X_train : numpy array
            Training data
        X_val : numpy array
            Validation data
        alpha_range : tuple
            Range for alpha search
            
        Returns:
        --------
        alpha_optimal : float
            Optimal regularization parameter
        """
        E_train = self.compute_sample_correlation(X_train)
        E_val = self.compute_sample_correlation(X_val)
        N_val = X_val.shape[0]
        
        def neg_log_likelihood(alpha):
            if self.method == 'identity_shrinkage':
                C = self.identity_shrinkage(E_train, alpha)
            elif self.method == 'group_shrinkage':
                C = self.group_shrinkage(E_train, alpha)
            elif self.method == 'graphical_lasso':
                C = self.graphical_lasso_reg(E_train, alpha)
            else:
                C = E_train
                
            return -self.log_likelihood(C, E_val, N_val)
        
        # Optimize alpha
        result = minimize_scalar(neg_log_likelihood, bounds=alpha_range, 
                                method='bounded')
        
        return result.x
    
    def compute_precision_matrix(self, C):
        """
        Compute precision matrix J = C^{-1}
        """
        # Add small regularization for numerical stability
        min_eig = np.linalg.eigvalsh(C).min()
        
        if min_eig < 1e-10:
            ridge = max(1e-6, abs(min_eig) * 1.1)
            print(f"  Adding ridge regularization: {ridge:.2e}")
            C_reg = C + ridge * np.eye(C.shape[0])
        else:
            C_reg = C
        
        try:
            J = linalg.inv(C_reg)
            
            # Check result
            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                raise linalg.LinAlgError("Inverse contains NaN/Inf")
                
            return J
        except linalg.LinAlgError:
            # More aggressive regularization
            ridge = 0.01
            print(f"  Warning: Using larger ridge regularization: {ridge}")
            C_reg = C + ridge * np.eye(C.shape[0])
            J = linalg.inv(C_reg)
            return J
    
    def compute_partial_correlation(self, J):
        """
        Phase III: Derive regularized partial correlation matrix
        
        Standardize precision matrix elements:
        J̃_ij = J_ij / sqrt(J_ii * J_jj)
        """
        diag = np.sqrt(np.abs(np.diag(J)))  # Use abs to handle numerical issues
        
        # Avoid division by zero
        diag = np.where(diag < 1e-10, 1e-10, diag)
        
        J_tilde = -J / np.outer(diag, diag)
        np.fill_diagonal(J_tilde, 1.0)
        
        # Clip to valid correlation range
        J_tilde = np.clip(J_tilde, -1, 1)
        
        return J_tilde
    
    def fit(self, X):
        """
        Complete regularization pipeline
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Raw data matrix (N samples × T features)
            
        Returns:
        --------
        self : NetworkRegularizer
            Fitted regularizer with computed matrices
        """
        # Phase I: Prepare data
        X_std = self.prepare_data(X)
        
        # Compute sample correlation
        self.sample_corr = self.compute_sample_correlation(X_std)
        
        # For partial correlation method, skip regularization
        if self.method == 'partial_correlation':
            print("\nUsing Partial Correlation (no regularization)")
            self.C_mu = self.sample_corr
            self.alpha_optimal = None
        else:
            # Split for validation
            X_train, X_val = train_test_split(X_std, test_size=self.test_size, 
                                              random_state=self.random_state)
            
            # Phase II: Find optimal alpha and regularize
            if self.method == 'ledoit_wolf':
                # Ledoit-Wolf doesn't require alpha optimization
                self.C_mu = self.ledoit_wolf_reg(X_std)
                self.alpha_optimal = None
            else:
                self.alpha_optimal = self.find_optimal_alpha(X_train, X_val)
                print(f"Optimal α = {self.alpha_optimal:.6f}")
                
                # Compute regularized correlation with full data
                E_full = self.compute_sample_correlation(X_std)
                
                if self.method == 'identity_shrinkage':
                    self.C_mu = self.identity_shrinkage(E_full, self.alpha_optimal)
                elif self.method == 'group_shrinkage':
                    self.C_mu = self.group_shrinkage(E_full, self.alpha_optimal)
                elif self.method == 'graphical_lasso':
                    self.C_mu = self.graphical_lasso_reg(E_full, self.alpha_optimal)
        
        # Compute precision matrix
        self.J_mu = self.compute_precision_matrix(self.C_mu)
        
        # Phase III: Compute partial correlations
        self.J_tilde = self.compute_partial_correlation(self.J_mu)
        
        # Set final output matrix based on method
        if self.method == 'partial_correlation':
            self.final_matrix = self.J_tilde
            print("Final output: Partial Correlation Matrix")
        elif self.method == 'graphical_lasso':
            self.final_matrix = self.C_mu
            print("Final output: GLASSO Regularized Correlation Matrix")
        else:
            self.final_matrix = self.C_mu
            print(f"Final output: {self.method.replace('_', ' ').title()} Regularized Correlation Matrix")
        
        return self
    
    def get_adjacency_matrix(self, threshold=None, absolute=True, as_dataframe=True):
        """
        Get sparse adjacency matrix for network graph
        
        Parameters:
        -----------
        threshold : float or None
            Percolation threshold for sparsification
            If None, returns full matrix
        absolute : bool
            Take absolute values of partial correlations
        as_dataframe : bool
            If True, return as pandas DataFrame with labels
            
        Returns:
        --------
        adj_matrix : numpy array or pandas DataFrame
            Adjacency matrix for functional network
        """
        if self.J_tilde is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        adj = np.abs(self.J_tilde) if absolute else self.J_tilde.copy()
        np.fill_diagonal(adj, 0)
        
        if threshold is not None:
            adj[adj < threshold] = 0
        
        if as_dataframe:
            return pd.DataFrame(adj, 
                              index=self.feature_names, 
                              columns=self.feature_names)
        return adj
    
    def get_matrix(self, matrix_type='final', as_dataframe=True):
        """
        Get the requested matrix
        
        Parameters:
        -----------
        matrix_type : str
            'final' - returns the main output matrix (default)
            'sample_corr' - raw sample correlation
            'regularized_corr' - regularized correlation (C_mu)
            'precision' - precision matrix (J_mu)
            'partial_corr' - partial correlation (J_tilde)
        as_dataframe : bool
            If True, return as pandas DataFrame with labels
            
        Returns:
        --------
        matrix : numpy array or pandas DataFrame
        """
        if self.final_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        matrix_map = {
            'final': self.final_matrix,
            'sample_corr': self.sample_corr,
            'regularized_corr': self.C_mu,
            'precision': self.J_mu,
            'partial_corr': self.J_tilde
        }
        
        if matrix_type not in matrix_map:
            raise ValueError(f"Invalid matrix_type. Choose from: {list(matrix_map.keys())}")
        
        matrix = matrix_map[matrix_type]
        
        if as_dataframe:
            return pd.DataFrame(matrix, 
                              index=self.feature_names, 
                              columns=self.feature_names)
        return matrix
    
    def summary(self):
        """Print summary of regularization results"""
        print("=" * 70)
        print("Network Regularization Summary")
        print("=" * 70)
        print(f"Method: {self.method}")
        print(f"Correlation axis: {self.correlation_axis}")
        if self.alpha_optimal is not None:
            print(f"Optimal α: {self.alpha_optimal:.6f}")
        print(f"\nMatrix dimensions: {self.final_matrix.shape[0]} × {self.final_matrix.shape[1]}")
        
        if self.correlation_axis == 'columns':
            print(f"Computing correlations between {self.final_matrix.shape[0]} features")
        else:
            print(f"Computing correlations between {self.final_matrix.shape[0]} samples")
            
        print(f"\nSample correlation range: [{self.sample_corr.min():.3f}, {self.sample_corr.max():.3f}]")
        print(f"Regularized correlation range: [{self.C_mu.min():.3f}, {self.C_mu.max():.3f}]")
        print(f"Partial correlation range: [{self.J_tilde.min():.3f}, {self.J_tilde.max():.3f}]")
        print(f"Final matrix range: [{self.final_matrix.min():.3f}, {self.final_matrix.max():.3f}]")
        
        # Sparsity analysis
        adj = np.abs(self.J_tilde.copy())
        np.fill_diagonal(adj, 0)
        density = (adj > 0).sum() / (adj.size - adj.shape[0])
        print(f"\nPartial correlation network density: {density:.3f}")
        
        # Show what matrix is returned
        print(f"\nOutput matrix type: ", end="")
        if self.method == 'partial_correlation':
            print("Partial Correlation (J̃)")
        elif self.method == 'graphical_lasso':
            print("GLASSO Regularized Correlation (C_μ)")
        else:
            print(f"{self.method.replace('_', ' ').title()} Regularized Correlation (C_μ)")
        print("=" * 70)


# Utility functions for network analysis
# Utility functions for network analysis
def diagnose_data(X):
    """
    Diagnose data quality issues
    
    Parameters:
    -----------
    X : pandas DataFrame or numpy array
        Input data to diagnose
    """
    if isinstance(X, pd.DataFrame):
        data = X.values
        cols = X.columns
    else:
        data = X
        cols = [f"Col_{i}" for i in range(X.shape[1])]
    
    print("="*60)
    print("DATA QUALITY DIAGNOSIS")
    print("="*60)
    print(f"Shape: {data.shape} (samples × features)")
    print(f"\nMissing values:")
    print(f"  Total NaN: {np.isnan(data).sum()}")
    print(f"  Total Inf: {np.isinf(data).sum()}")
    
    # Per column analysis
    nan_per_col = np.isnan(data).sum(axis=0)
    inf_per_col = np.isinf(data).sum(axis=0)
    
    if nan_per_col.sum() > 0:
        print(f"\n  Columns with NaN:")
        for i, (col, count) in enumerate(zip(cols, nan_per_col)):
            if count > 0:
                print(f"    {col}: {count} ({count/data.shape[0]*100:.1f}%)")
    
    if inf_per_col.sum() > 0:
        print(f"\n  Columns with Inf:")
        for i, (col, count) in enumerate(zip(cols, inf_per_col)):
            if count > 0:
                print(f"    {col}: {count}")
    
    # Zero variance check
    col_std = np.nanstd(data, axis=0)
    zero_var = col_std < 1e-10
    if zero_var.sum() > 0:
        print(f"\n  Zero-variance columns: {zero_var.sum()}")
        for col, is_zero in zip(cols, zero_var):
            if is_zero:
                print(f"    {col}")
    
    print("\nData statistics:")
    print(f"  Min: {np.nanmin(data):.6f}")
    print(f"  Max: {np.nanmax(data):.6f}")
    print(f"  Mean: {np.nanmean(data):.6f}")
    print(f"  Std: {np.nanstd(data):.6f}")
    print("="*60)


def compute_newman_modularity(adj_matrix, communities):
    """
    Compute Newman modularity Q
    
    Parameters:
    -----------
    adj_matrix : numpy array
        Adjacency matrix
    communities : numpy array
        Community assignments for each node
        
    Returns:
    --------
    Q : float
        Modularity score
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    
    A = adj_matrix.copy()
    np.fill_diagonal(A, 0)
    
    m = A.sum() / 2  # Total edge weight
    if m == 0:
        return 0
    
    k = A.sum(axis=1)  # Degree
    
    Q = 0
    unique_communities = np.unique(communities)
    
    for c in unique_communities:
        nodes_in_c = communities == c
        
        # Edges within community
        l_c = A[np.ix_(nodes_in_c, nodes_in_c)].sum() / 2
        
        # Expected edges
        d_c = k[nodes_in_c].sum()
        
        Q += l_c / m - (d_c / (2 * m)) ** 2
    
    return Q


def compute_nmi(communities1, communities2):
    """
    Compute Normalized Mutual Information between two community assignments
    
    Parameters:
    -----------
    communities1, communities2 : numpy array
        Community assignments
        
    Returns:
    --------
    nmi : float
        NMI score [0, 1]
    """
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(communities1, communities2)


# --------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
import networkx as nx

def compute_partial_corr_robust(X, feature_names=None, alpha_range=(1e-2, 1.0), n_alphas=10):
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns if feature_names is None else feature_names
        X = X.values
    elif feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Remove constant columns
    X_std = X.std(axis=0)
    nonconstant_cols = np.where(X_std > 0)[0]
    X = X[:, nonconstant_cols]
    feature_names = [feature_names[i] for i in nonconstant_cols]

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    try:
        model = GraphicalLassoCV(alphas=np.linspace(alpha_range[0], alpha_range[1], n_alphas), assume_centered=False)
        model.fit(X_standardized)
        C_mu, J_mu = model.covariance_, model.precision_
        method = "glasso"
        alpha_L = model.alpha_
    except FloatingPointError:
        model = LedoitWolf()
        model.fit(X_standardized)
        C_mu, J_mu = model.covariance_, model.precision_
        method = "ledoitwolf"
        alpha_L = None

    d = np.sqrt(np.diag(J_mu))
    tilde_J = J_mu / np.outer(d, d)
    np.fill_diagonal(tilde_J, 1)

    tilde_J_df = pd.DataFrame(tilde_J, index=feature_names, columns=feature_names)
    return tilde_J_df, alpha_L, C_mu, J_mu, method

def fully_connected_thresholded_matrix(tilde_J_df):
    """
    Increases threshold for absolute edge weight until the graph just starts to disconnect.
    Returns the largest threshold matrix that is still fully connected.
    """
    abs_weights = np.abs(tilde_J_df.values)
    n_nodes = tilde_J_df.shape[0]
    abs_weights[range(n_nodes), range(n_nodes)] = 0 # zero out diagonal

    # Get all unique non-diagonal weights (upper triangle)
    weights = abs_weights[np.triu_indices(n_nodes, k=1)]
    thresholds = np.unique(np.sort(weights)) # ascending order

    # Iterate over thresholds, keep lowest ones first
    final_matrix = tilde_J_df.values.copy()
    final_thresh = 0
    for thresh in thresholds:
        mask = abs_weights >= thresh
        temp_matrix = tilde_J_df.values * mask

        # Build graph
        G = nx.Graph()
        nodes = tilde_J_df.index.tolist()
        G.add_nodes_from(nodes)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if mask[i, j]:
                    G.add_edge(nodes[i], nodes[j], weight=temp_matrix[i, j])

        if nx.is_connected(G): # still connected, keep going
            final_matrix = temp_matrix.copy()
            final_thresh = thresh
        else: # disconnected, stop at previous threshold
            break

    thresholded_df = pd.DataFrame(final_matrix, index=tilde_J_df.index, columns=tilde_J_df.columns)
    return final_thresh, thresholded_df

