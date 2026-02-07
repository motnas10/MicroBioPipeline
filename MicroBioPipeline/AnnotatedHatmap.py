# ---------------------------------------------------------------------------------------------------------------
# Annotated Heatmap with Row/Column Categories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap, Colormap, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from typing import Optional, Dict, Tuple, Union, Literal, List, Sequence, Callable


def _is_numeric_column(series):
    """Check if a pandas Series contains numeric data."""
    return pd.api.types.is_numeric_dtype(series) and not series.dropna().empty


def _get_text_color_for_background(background_color, threshold=0.5):
    """
    Determine if text should be black or white based on background color luminance.
    
    Parameters
    ----------
    background_color : array-like
        RGBA color array.
    threshold : float
        Luminance threshold (0-1). Below this, use white text.
    
    Returns
    -------
    str
        'white' or 'black'
    """
    # Calculate relative luminance
    r, g, b = background_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < threshold else 'black'


def plot_annotated_heatmap(
    # -------------------------------------------------------------------------
    # 1. CORE DATA & FIGURE SETUP
    # -------------------------------------------------------------------------
    data: pd.DataFrame,
    transpose: bool = False,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    ax_size: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 14),
    square: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 600,
    font_scale: float = 13,
    font_size_func: Optional[Callable] = None,
    # -------------------------------------------------------------------------
    # 2. HEATMAP STYLING & COLORS (Main Data)
    # -------------------------------------------------------------------------
    heatmap_type: Literal['qualitative', 'quantitative'] = 'qualitative',
    cmap: Union[str, LinearSegmentedColormap] = 'Blues',
    # Numeric Data Control
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    robust: bool = False,
    # Categorical Data Control
    categorical_data: bool = False,
    categorical_palette: Optional[Dict[str, str]] = None,
    categorical_legend_labels: Optional[Dict[str, str]] = None,
    # Grid Styling
    heatmap_linewidths: float = 0.5,
    heatmap_linecolor: str = 'grey',
    # -------------------------------------------------------------------------
    # 3. CELL ANNOTATIONS (Text Inside Cells)
    # -------------------------------------------------------------------------
    show_cell_annotations: bool = False,
    cell_annotation_data: Optional[pd.DataFrame] = None,
    cell_annotation_format: str = '{x}',
    cell_annotation_fontsize: Optional[float] = None,
    cell_annotation_color: Union[str, Literal['auto']] = 'auto',
    cell_annotation_threshold: Optional[float] = None,
    cell_annotation_weight: str = 'normal',
    # -------------------------------------------------------------------------
    # 4. ROW ANNOTATIONS (Sidebars)
    # -------------------------------------------------------------------------
    row_annotations: Optional[pd.DataFrame] = None,
    row_annotation_col: Optional[Union[str, List[str]]] = None,
    row_palette: Optional[Union[Dict[str, str], List[Union[Dict[str, str], str, Colormap]]]] = None,
    # Sizing
    row_patch_width: Optional[Union[float, List[float]]] = None,
    row_patch_auto_width: bool = True,
    patch_width_ratio: float = 0.05,
    row_patch_spacing: float = 0.0,
    row_patch_alpha: float = 1.0,
    row_patch_offset: float = 0.1,
    # Normalization for Continuous Annotations
    row_vmin: Optional[Union[float, List[float]]] = None,
    row_vmax: Optional[Union[float, List[float]]] = None,
    row_center: Optional[Union[float, List[float]]] = None,
    # -------------------------------------------------------------------------
    # 5. COLUMN ANNOTATIONS (Top Bars)
    # -------------------------------------------------------------------------
    col_annotations: Optional[pd.DataFrame] = None,
    col_annotation_col: Optional[Union[str, List[str]]] = None,
    col_palette: Optional[Union[Dict[str, str], List[Union[Dict[str, str], str, Colormap]]]] = None,
    # Sizing
    col_patch_height: Optional[Union[float, List[float]]] = None,
    col_patch_auto_height: bool = True,
    patch_height_ratio: float = 0.05,
    col_patch_spacing: float = 0.0,
    col_patch_alpha: float = 1.0,
    col_patch_offset: float = 0.1,
    # Normalization for Continuous Annotations
    col_vmin: Optional[Union[float, List[float]]] = None,
    col_vmax: Optional[Union[float, List[float]]] = None,
    col_center: Optional[Union[float, List[float]]] = None,
    # -------------------------------------------------------------------------
    # 6. SIGNIFICANCE OVERLAY (Markers/Stars)
    # -------------------------------------------------------------------------
    significance_data: Optional[pd.DataFrame] = None,
    significance_func: Optional[Callable] = None,
    significance_marker: Literal['circle', 'star', 'asterisk', 'text'] = 'circle',
    significance_size_map: Optional[Dict[str, float]] = None,
    significance_color: str = 'black',
    significance_alpha: float = 1.0,
    significance_linewidth: float = 0.5,
    significance_edgecolor: Optional[str] = None,
    significance_text_size: Optional[float] = None,
    circle_background: str = 'white',
    # -------------------------------------------------------------------------
    # 7. SEPARATORS & BORDERS
    # -------------------------------------------------------------------------
    # Row Separators
    row_separation_col: Optional[Union[str, List[str]]] = None,
    row_separation_linewidth: Union[float, List[float]] = 2.0,
    row_separation_color: Union[str, List[str]] = 'black',
    row_separation_linestyle: Union[str, List[str]] = '-',
    row_separation_alpha: Union[float, List[float]] = 1.0,
    # Column Separators
    col_separation_col: Optional[Union[str, List[str]]] = None,
    col_separation_linewidth: Union[float, List[float]] = 2.0,
    col_separation_color: Union[str, List[str]] = 'black',
    col_separation_linestyle: Union[str, List[str]] = '-',
    col_separation_alpha: Union[float, List[float]] = 1.0,
    # Patch Borders
    patch_linewidths: float = 0.0,
    patch_linecolor: str = 'grey',    
    # -------------------------------------------------------------------------
    # 8. AXES LABELS & TICKS
    # -------------------------------------------------------------------------
    title: str = 'Annotated Heatmap',
    xlabel: str = 'Columns',
    ylabel: str = 'Rows',
    # X Ticks
    xticklabels: Optional[Union[pd.Series, List, np.ndarray]] = None,
    xticklabels_rotation: float = 45,
    xticklabels_position: Literal['top', 'bottom'] = 'bottom',
    # Y Ticks
    yticklabels: Optional[Union[pd.Series, List, np.ndarray]] = None,
    yticklabels_rotation: float = 0,
    yticklabels_position: Literal['left', 'right'] = 'left',
    # Padding
    auto_tick_padding: bool = True,
    tick_pad_x: Optional[float] = None,
    tick_pad_y: Optional[float] = None,
    tick_pad_ratio: float = 1.5,
    base_tick_pad: float = 5.0,
    # -------------------------------------------------------------------------
    # 9. MAIN COLORBAR (Heatmap Values)
    # -------------------------------------------------------------------------
    show_colorbar: bool = False,
    colorbar_position: Literal['right', 'left', 'top', 'bottom'] = 'right',
    colorbar_size: str = '5%',
    colorbar_pad: float = 0.05,
    colorbar_label: Optional[str] = None,
    cbar_ticks: Optional[Sequence[float]] = None,
    colorbar_orientation: Optional[Literal['vertical', 'horizontal']] = None,
    colorbar_coords: Optional[Tuple[float, float, float, float]] = None,
    colorbar_ax: Optional[plt.Axes] = None,
    separate_colorbar: bool = False,
    colorbar_figsize: Optional[Tuple[float, float]] = None,
    cbar_kws: Optional[Dict] = None,
    # -------------------------------------------------------------------------
    # 10. ANNOTATION COLORBARS (Rows)
    # -------------------------------------------------------------------------
    row_colorbar_position: Optional[Union[str, List[str]]] = None,
    row_colorbar_size: Optional[Union[str, List[str]]] = None,
    row_colorbar_pad: Optional[Union[float, List[float]]] = None,
    row_colorbar_label: Optional[Union[str, List[str]]] = None,
    row_cbar_ticks: Optional[Union[Sequence[float], List[Sequence[float]]]] = None,
    row_colorbar_orientation: Optional[Union[str, List[str]]] = None,
    row_colorbar_coords: Optional[Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]] = None,
    row_colorbar_ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
    row_separate_colorbar: Optional[Union[bool, List[bool]]] = None,
    row_colorbar_figsize: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    # -------------------------------------------------------------------------
    # 11. ANNOTATION COLORBARS (Columns)
    # -------------------------------------------------------------------------
    col_colorbar_position: Optional[Union[str, List[str]]] = None,
    col_colorbar_size: Optional[Union[str, List[str]]] = None,
    col_colorbar_pad: Optional[Union[float, List[float]]] = None,
    col_colorbar_label: Optional[Union[str, List[str]]] = None,
    col_cbar_ticks: Optional[Union[Sequence[float], List[Sequence[float]]]] = None,
    col_colorbar_orientation: Optional[Union[str, List[str]]] = None,
    col_colorbar_coords: Optional[Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]] = None,
    col_colorbar_ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
    col_separate_colorbar: Optional[Union[bool, List[bool]]] = None,
    col_colorbar_figsize: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    # -------------------------------------------------------------------------
    # 12. LEGENDS
    # -------------------------------------------------------------------------
    # Titles & Labels
    row_legend_title: Union[str, List[str]] = 'Row Categories',
    col_legend_title: Union[str, List[str]] = 'Column Categories',
    value_legend_title: str = 'Values',
    value_legend_labels: Optional[Dict] = None,
    show_significance_legend: bool = True,
    significance_legend_title: str = 'Significance',
    # Positioning & Styling
    legend_position: Literal['right', 'left', 'top', 'bottom'] = 'right',
    legend_alignment: Literal['top', 'center', 'bottom'] = 'top',
    legend_bbox_x: float = 1.02,
    legend_auto_spacing: bool = True,
    legend_spacing: float = 0.08,
    legend_order: Optional[list] = None,
    separate_legends: bool = False,
    legend_orientation: Literal['vertical', 'horizontal'] = 'vertical',
    legend_patch_linewidths: Optional[float] = 0.5,
    legend_patch_linecolor: Optional[str] = 'gray',
) -> Tuple[plt.Figure, plt.Axes, Dict, Optional[plt.Figure], Dict]:
    """
    Create an annotated heatmap with colored patches for row and column categories. 
    Supports both qualitative (binary/categorical) and quantitative (gradient) heatmaps.
    Now supports significance overlays, quantitative patch annotations, categorical data, and cell annotations.
    
    Parameters
    ----------
    data : pd.DataFrame
        The main data to plot as a heatmap.
    transpose : bool, default False
        If True, transpose the data matrix and swap all row/column related parameters.
    [... previous parameters ...]
    categorical_data : bool, default False
        If True, treat main heatmap data as categorical (not quantitative).
        Data values will be mapped to colors using categorical_palette.
    categorical_palette : dict, optional
        Dictionary mapping category values to colors.
        Required when categorical_data=True.
        Keys should match values in data DataFrame.
        Example: {'low': 'blue', 'medium': 'yellow', 'high': 'red'}
    categorical_legend_labels : dict, optional
        Dictionary mapping category values to display labels in legend.
        If None, uses category values as labels.
        Example: {'low': 'Low Expression', 'medium': 'Medium', 'high': 'High'}
    show_cell_annotations : bool, default False
        If True, show text annotations in each cell.
    cell_annotation_data : pd.DataFrame, optional
        DataFrame with same shape as data containing values to display in cells.
        If None and show_cell_annotations=True:
            - For categorical_data=True: displays the category values from data
            - For categorical_data=False: displays the numeric values from data
    cell_annotation_format : str, default '{x}'
        Format string for cell annotations. Use {x} for value.
        Examples: '{x}', '{x:.2f}', '{x:.1%}'
        Only applies when cell_annotation_data contains numeric values.
    cell_annotation_fontsize : float, optional
        Font size for cell annotations. If None, auto-calculated.
    cell_annotation_color : str or 'auto', default 'auto'
        Color for cell annotations. If 'auto', uses black/white based on background.
    cell_annotation_threshold : float, optional
        Luminance threshold for auto color selection (0-1).
        If None, uses 0.5.
    cell_annotation_weight : str, default 'normal'
        Font weight for cell annotations ('normal', 'bold', etc.).
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The main figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    legend_figs : dict
        Dictionary of legend figure objects.
    colorbar_fig : matplotlib.figure.Figure or None
        Separate colorbar figure if separate_colorbar=True.
    patch_colorbar_figs : dict
        Dictionary of separate patch colorbar figures.
    
    Examples
    --------
    >>> # Categorical heatmap with annotations
    >>> data_cat = pd.DataFrame({
    ...     'S1': ['Low', 'Med', 'High'],
    ...     'S2': ['High', 'Low', 'Med']
    ... })
    >>> cat_palette = {'Low': 'blue', 'Med': 'yellow', 'High': 'red'}
    >>> fig, ax, legs, cbar, patch_cbars = plot_annotated_heatmap(
    ...     data=data_cat,
    ...     categorical_data=True,
    ...     categorical_palette=cat_palette,
    ...     show_cell_annotations=True,
    ...     cell_annotation_color='white',
    ...     separate_legends=True
    ... )
    """
    
    # Set defaults for legend patch styling
    if legend_patch_linewidths is None:
        legend_patch_linewidths = patch_linewidths
    if legend_patch_linecolor is None:
        legend_patch_linecolor = patch_linecolor
    
    # Validate fig and ax parameters
    if (fig is None) != (ax is None):
        raise ValueError("Both fig and ax must be provided together, or both must be None")
    
    # Determine if we're using existing axes
    use_existing_ax = (fig is not None and ax is not None)

    if transpose:
        # Transpose main data
        data = data.T
        
        # Transpose significance data if provided
        if significance_data is not None:
            significance_data = significance_data.T
        
        # Transpose cell annotation data if provided
        if cell_annotation_data is not None:
            cell_annotation_data = cell_annotation_data.T

        # Swap annotations
        row_annotations, col_annotations = col_annotations, row_annotations
        row_annotation_col, col_annotation_col = col_annotation_col, row_annotation_col
        row_palette, col_palette = col_palette, row_palette
        row_vmin, col_vmin = col_vmin, row_vmin
        row_vmax, col_vmax = col_vmax, row_vmax
        row_center, col_center = col_center, row_center
        
        # Swap colorbar parameters
        row_colorbar_position, col_colorbar_position = col_colorbar_position, row_colorbar_position
        row_colorbar_size, col_colorbar_size = col_colorbar_size, row_colorbar_size
        row_colorbar_pad, col_colorbar_pad = col_colorbar_pad, row_colorbar_pad
        row_colorbar_label, col_colorbar_label = col_colorbar_label, row_colorbar_label
        row_cbar_ticks, col_cbar_ticks = col_cbar_ticks, row_cbar_ticks
        row_colorbar_orientation, col_colorbar_orientation = col_colorbar_orientation, row_colorbar_orientation
        row_colorbar_coords, col_colorbar_coords = col_colorbar_coords, row_colorbar_coords
        row_colorbar_ax, col_colorbar_ax = col_colorbar_ax, row_colorbar_ax
        row_separate_colorbar, col_separate_colorbar = col_separate_colorbar, row_separate_colorbar
        row_colorbar_figsize, col_colorbar_figsize = col_colorbar_figsize, row_colorbar_figsize

        # Swap labels and titles
        xlabel, ylabel = ylabel, xlabel
        row_legend_title, col_legend_title = col_legend_title, row_legend_title

        # Swap patch dimensions
        row_patch_width, col_patch_height = col_patch_height, row_patch_width
        row_patch_auto_width, col_patch_auto_height = col_patch_auto_height, row_patch_auto_width
        row_patch_spacing, col_patch_spacing = col_patch_spacing, row_patch_spacing
        row_patch_alpha, col_patch_alpha = col_patch_alpha, row_patch_alpha
        row_patch_offset, col_patch_offset = col_patch_offset, row_patch_offset

        # Swap tick labels and rotations
        xticklabels, yticklabels = yticklabels, xticklabels
        xticklabels_rotation, yticklabels_rotation = yticklabels_rotation, xticklabels_rotation
        tick_pad_x, tick_pad_y = tick_pad_y, tick_pad_x

        # Swap separation logic
        row_separation_col, col_separation_col = col_separation_col, row_separation_col
        row_separation_linewidth, col_separation_linewidth = (
            col_separation_linewidth,
            row_separation_linewidth,
        )
        row_separation_color, col_separation_color = col_separation_color, row_separation_color
        row_separation_linestyle, col_separation_linestyle = (
            col_separation_linestyle,
            row_separation_linestyle,
        )
        row_separation_alpha, col_separation_alpha = col_separation_alpha, row_separation_alpha

    # Validate significance data if provided
    if significance_data is not None:
        assert significance_data.shape == data.shape, "significance_data must match data shape"
        assert (significance_data.index == data.index).all(), "significance_data index must match data index"
        assert (significance_data.columns == data.columns).all(), "significance_data columns must match data columns"

    # Validate categorical data parameters
    if categorical_data:
        assert categorical_palette is not None, "categorical_palette is required when categorical_data=True"
        
        # Verify all data values are in the palette
        unique_values = pd.unique(data.values.ravel())
        unique_values = unique_values[pd.notna(unique_values)]
        missing_categories = set(unique_values) - set(categorical_palette.keys())
        if missing_categories:
            raise ValueError(f"Categories {missing_categories} not found in categorical_palette")
        
        # Validate categorical_legend_labels if provided
        if categorical_legend_labels is not None:
            missing_labels = set(categorical_palette.keys()) - set(categorical_legend_labels.keys())
            if missing_labels:
                raise ValueError(f"Categories {missing_labels} not found in categorical_legend_labels")
    
    # Validate cell annotation data
    if cell_annotation_data is not None:
        assert cell_annotation_data.shape == data.shape, "cell_annotation_data must have same shape as data"
        assert (cell_annotation_data.index == data.index).all(), "cell_annotation_data index must match data"
        assert (cell_annotation_data.columns == data.columns).all(), "cell_annotation_data columns must match data"

    # Convert single values to lists for uniform handling
    if row_annotation_col is not None and not isinstance(row_annotation_col, list):
        row_annotation_col = [row_annotation_col]
    if col_annotation_col is not None and not isinstance(col_annotation_col, list):
        col_annotation_col = [col_annotation_col]
    if row_palette is not None and not isinstance(row_palette, list):
        row_palette = [row_palette]
    if col_palette is not None and not isinstance(col_palette, list):
        col_palette = [col_palette]
    if isinstance(row_legend_title, str):
        row_legend_title = [row_legend_title]
    if isinstance(col_legend_title, str):
        col_legend_title = [col_legend_title]
    if row_patch_width is not None and not isinstance(row_patch_width, list):
        row_patch_width = [row_patch_width]
    if col_patch_height is not None and not isinstance(col_patch_height, list):
        col_patch_height = [col_patch_height]
    
    # Convert vmin/vmax/center to lists
    if row_vmin is not None and not isinstance(row_vmin, list):
        row_vmin = [row_vmin]
    if row_vmax is not None and not isinstance(row_vmax, list):
        row_vmax = [row_vmax]
    if row_center is not None and not isinstance(row_center, list):
        row_center = [row_center]
    if col_vmin is not None and not isinstance(col_vmin, list):
        col_vmin = [col_vmin]
    if col_vmax is not None and not isinstance(col_vmax, list):
        col_vmax = [col_vmax]
    if col_center is not None and not isinstance(col_center, list):
        col_center = [col_center]
    
    # Convert row colorbar parameters to lists
    if row_colorbar_position is not None and not isinstance(row_colorbar_position, list):
        row_colorbar_position = [row_colorbar_position]
    if row_colorbar_size is not None and not isinstance(row_colorbar_size, list):
        row_colorbar_size = [row_colorbar_size]
    if row_colorbar_pad is not None and not isinstance(row_colorbar_pad, list):
        row_colorbar_pad = [row_colorbar_pad]
    if row_colorbar_label is not None and not isinstance(row_colorbar_label, list):
        row_colorbar_label = [row_colorbar_label]
    if row_cbar_ticks is not None and not isinstance(row_cbar_ticks, list):
        row_cbar_ticks = [row_cbar_ticks]
    if row_colorbar_orientation is not None and not isinstance(row_colorbar_orientation, list):
        row_colorbar_orientation = [row_colorbar_orientation]
    if row_colorbar_coords is not None and not isinstance(row_colorbar_coords, list):
        row_colorbar_coords = [row_colorbar_coords]
    if row_colorbar_ax is not None and not isinstance(row_colorbar_ax, list):
        row_colorbar_ax = [row_colorbar_ax]
    if row_separate_colorbar is not None and not isinstance(row_separate_colorbar, list):
        row_separate_colorbar = [row_separate_colorbar]
    if row_colorbar_figsize is not None and not isinstance(row_colorbar_figsize, list):
        row_colorbar_figsize = [row_colorbar_figsize]
    
    # Convert col colorbar parameters to lists
    if col_colorbar_position is not None and not isinstance(col_colorbar_position, list):
        col_colorbar_position = [col_colorbar_position]
    if col_colorbar_size is not None and not isinstance(col_colorbar_size, list):
        col_colorbar_size = [col_colorbar_size]
    if col_colorbar_pad is not None and not isinstance(col_colorbar_pad, list):
        col_colorbar_pad = [col_colorbar_pad]
    if col_colorbar_label is not None and not isinstance(col_colorbar_label, list):
        col_colorbar_label = [col_colorbar_label]
    if col_cbar_ticks is not None and not isinstance(col_cbar_ticks, list):
        col_cbar_ticks = [col_cbar_ticks]
    if col_colorbar_orientation is not None and not isinstance(col_colorbar_orientation, list):
        col_colorbar_orientation = [col_colorbar_orientation]
    if col_colorbar_coords is not None and not isinstance(col_colorbar_coords, list):
        col_colorbar_coords = [col_colorbar_coords]
    if col_colorbar_ax is not None and not isinstance(col_colorbar_ax, list):
        col_colorbar_ax = [col_colorbar_ax]
    if col_separate_colorbar is not None and not isinstance(col_separate_colorbar, list):
        col_separate_colorbar = [col_separate_colorbar]
    if col_colorbar_figsize is not None and not isinstance(col_colorbar_figsize, list):
        col_colorbar_figsize = [col_colorbar_figsize]
    
    # Convert separation parameters to lists
    if row_separation_col is not None and not isinstance(row_separation_col, list):
        row_separation_col = [row_separation_col]
    if col_separation_col is not None and not isinstance(col_separation_col, list):
        col_separation_col = [col_separation_col]
    if row_separation_col is not None: 
        if not isinstance(row_separation_linewidth, list):
            row_separation_linewidth = [row_separation_linewidth] * len(row_separation_col)
        if not isinstance(row_separation_color, list):
            row_separation_color = [row_separation_color] * len(row_separation_col)
        if not isinstance(row_separation_linestyle, list):
            row_separation_linestyle = [row_separation_linestyle] * len(row_separation_col)
        if not isinstance(row_separation_alpha, list):
            row_separation_alpha = [row_separation_alpha] * len(row_separation_col)
    if col_separation_col is not None:
        if not isinstance(col_separation_linewidth, list):
            col_separation_linewidth = [col_separation_linewidth] * len(col_separation_col)
        if not isinstance(col_separation_color, list):
            col_separation_color = [col_separation_color] * len(col_separation_col)
        if not isinstance(col_separation_linestyle, list):
            col_separation_linestyle = [col_separation_linestyle] * len(col_separation_col)
        if not isinstance(col_separation_alpha, list):
            col_separation_alpha = [col_separation_alpha] * len(col_separation_col)
    
    # Process custom tick labels
    if xticklabels is not None:  
        if isinstance(xticklabels, pd.Series):
            xticklabels = xticklabels.loc[data.columns].values
        else:
            xticklabels = list(xticklabels)
        assert len(xticklabels) == len(data.columns), "xticklabels must match data.columns length"
    else:
        xticklabels = data.columns
    
    if yticklabels is not None: 
        if isinstance(yticklabels, pd.Series):
            yticklabels = yticklabels.loc[data.index].values
        else:
            yticklabels = list(yticklabels)
        assert len(yticklabels) == len(data.index), "yticklabels must match data.index length"
    else:
        yticklabels = data.index
    
    # Validate list lengths for row annotations
    if row_annotation_col is not None: 
        n_row_annots = len(row_annotation_col)
        if row_palette is not None:
            assert len(row_palette) == n_row_annots, "row_palette must match row_annotation_col length"
        if len(row_legend_title) == 1:
            row_legend_title = row_legend_title * n_row_annots
        assert len(row_legend_title) == n_row_annots, "row_legend_title must match row_annotation_col length"
        if row_patch_width is not None:
            if len(row_patch_width) == 1:
                row_patch_width = row_patch_width * n_row_annots
            assert len(row_patch_width) == n_row_annots, "row_patch_width must match row_annotation_col length"
        if row_vmin is not None:
            if len(row_vmin) == 1:
                row_vmin = row_vmin * n_row_annots
            assert len(row_vmin) == n_row_annots, "row_vmin must match row_annotation_col length"
        if row_vmax is not None:
            if len(row_vmax) == 1:
                row_vmax = row_vmax * n_row_annots
            assert len(row_vmax) == n_row_annots, "row_vmax must match row_annotation_col length"
        if row_center is not None:
            if len(row_center) == 1:
                row_center = row_center * n_row_annots
            assert len(row_center) == n_row_annots, "row_center must match row_annotation_col length"
        
        # Expand row colorbar parameters to match annotation count
        if row_colorbar_position is not None:
            if len(row_colorbar_position) == 1:
                row_colorbar_position = row_colorbar_position * n_row_annots
        if row_colorbar_size is not None:
            if len(row_colorbar_size) == 1:
                row_colorbar_size = row_colorbar_size * n_row_annots
        if row_colorbar_pad is not None:
            if len(row_colorbar_pad) == 1:
                row_colorbar_pad = row_colorbar_pad * n_row_annots
        if row_colorbar_label is not None:
            if len(row_colorbar_label) == 1:
                row_colorbar_label = row_colorbar_label * n_row_annots
        if row_cbar_ticks is not None:
            if len(row_cbar_ticks) == 1:
                row_cbar_ticks = row_cbar_ticks * n_row_annots
        if row_colorbar_orientation is not None:
            if len(row_colorbar_orientation) == 1:
                row_colorbar_orientation = row_colorbar_orientation * n_row_annots
        if row_colorbar_coords is not None:
            if len(row_colorbar_coords) == 1:
                row_colorbar_coords = row_colorbar_coords * n_row_annots
        if row_colorbar_ax is not None:
            if len(row_colorbar_ax) == 1:
                row_colorbar_ax = row_colorbar_ax * n_row_annots
        if row_separate_colorbar is not None:
            if len(row_separate_colorbar) == 1:
                row_separate_colorbar = row_separate_colorbar * n_row_annots
        if row_colorbar_figsize is not None:
            if len(row_colorbar_figsize) == 1:
                row_colorbar_figsize = row_colorbar_figsize * n_row_annots
    
    # Validate list lengths for col annotations
    if col_annotation_col is not None:
        n_col_annots = len(col_annotation_col)
        if col_palette is not None:  
            assert len(col_palette) == n_col_annots, "col_palette must match col_annotation_col length"
        if len(col_legend_title) == 1:
            col_legend_title = col_legend_title * n_col_annots
        assert len(col_legend_title) == n_col_annots, "col_legend_title must match col_annotation_col length"
        if col_patch_height is not None:  
            if len(col_patch_height) == 1:
                col_patch_height = col_patch_height * n_col_annots
            assert len(col_patch_height) == n_col_annots, "col_patch_height must match col_annotation_col length"
        if col_vmin is not None:
            if len(col_vmin) == 1:
                col_vmin = col_vmin * n_col_annots
            assert len(col_vmin) == n_col_annots, "col_vmin must match col_annotation_col length"
        if col_vmax is not None:
            if len(col_vmax) == 1:
                col_vmax = col_vmax * n_col_annots
            assert len(col_vmax) == n_col_annots, "col_vmax must match col_annotation_col length"
        if col_center is not None:
            if len(col_center) == 1:
                col_center = col_center * n_col_annots
            assert len(col_center) == n_col_annots, "col_center must match col_annotation_col length"
        
        # Expand col colorbar parameters to match annotation count
        if col_colorbar_position is not None:
            if len(col_colorbar_position) == 1:
                col_colorbar_position = col_colorbar_position * n_col_annots
        if col_colorbar_size is not None:
            if len(col_colorbar_size) == 1:
                col_colorbar_size = col_colorbar_size * n_col_annots
        if col_colorbar_pad is not None:
            if len(col_colorbar_pad) == 1:
                col_colorbar_pad = col_colorbar_pad * n_col_annots
        if col_colorbar_label is not None:
            if len(col_colorbar_label) == 1:
                col_colorbar_label = col_colorbar_label * n_col_annots
        if col_cbar_ticks is not None:
            if len(col_cbar_ticks) == 1:
                col_cbar_ticks = col_cbar_ticks * n_col_annots
        if col_colorbar_orientation is not None:
            if len(col_colorbar_orientation) == 1:
                col_colorbar_orientation = col_colorbar_orientation * n_col_annots
        if col_colorbar_coords is not None:
            if len(col_colorbar_coords) == 1:
                col_colorbar_coords = col_colorbar_coords * n_col_annots
        if col_colorbar_ax is not None:
            if len(col_colorbar_ax) == 1:
                col_colorbar_ax = col_colorbar_ax * n_col_annots
        if col_separate_colorbar is not None:
            if len(col_separate_colorbar) == 1:
                col_separate_colorbar = col_separate_colorbar * n_col_annots
        if col_colorbar_figsize is not None:
            if len(col_colorbar_figsize) == 1:
                col_colorbar_figsize = col_colorbar_figsize * n_col_annots
    
    # Validate separation columns
    if row_separation_col is not None and row_annotations is not None:
        for sep_col in row_separation_col:
            assert sep_col in row_annotations.columns, f"row_separation_col '{sep_col}' not found in row_annotations"
        assert len(row_separation_linewidth) == len(row_separation_col), "row_separation_linewidth must match row_separation_col length"
        assert len(row_separation_color) == len(row_separation_col), "row_separation_color must match row_separation_col length"
        assert len(row_separation_linestyle) == len(row_separation_col), "row_separation_linestyle must match row_separation_col length"
        assert len(row_separation_alpha) == len(row_separation_col), "row_separation_alpha must match row_separation_col length"
    
    if col_separation_col is not None and col_annotations is not None: 
        for sep_col in col_separation_col:
            assert sep_col in col_annotations.columns, f"col_separation_col '{sep_col}' not found in col_annotations"
        assert len(col_separation_linewidth) == len(col_separation_col), "col_separation_linewidth must match col_separation_col length"
        assert len(col_separation_color) == len(col_separation_col), "col_separation_color must match col_separation_col length"
        assert len(col_separation_linestyle) == len(col_separation_col), "col_separation_linestyle must match col_separation_col length"
        assert len(col_separation_alpha) == len(col_separation_col), "col_separation_alpha must match col_separation_col length"
    
    # Determine size for font calculations
    if use_existing_ax:
        if ax_size is not None:
            calc_width, calc_height = ax_size
        else:
            bbox = ax.get_position()
            fig_width, fig_height = fig.get_size_inches()
            calc_width = bbox.width * fig_width
            calc_height = bbox.height * fig_height
    else:
        calc_width, calc_height = figsize
    
    # Auto-calculate cell annotation font size if not provided
    if show_cell_annotations and cell_annotation_fontsize is None:
        # Base it on figure size and data dimensions
        n_rows, n_cols = data.shape
        avg_cell_size = min(calc_width / n_cols, calc_height / n_rows)
        cell_annotation_fontsize = max(4, min(12, avg_cell_size * 72 * 0.3))  # Scale with cell size
    
    if cell_annotation_threshold is None:
        cell_annotation_threshold = 0.5
    
    # Calculate font sizes
    if font_size_func is not None:
        font_size = font_size_func(calc_width, calc_height, 'in', scale=font_scale)
    else:
        font_size = {
            'title': 20,
            'suptitle': 24,
            'axes_label': 16,
            'ticks_label': 14,
            'legend': 7,
            'legend_title': 8,
            'annotation': 8,
            'cbar_label': 12,
            'cbar_ticks': 10,
            'label': 10,
            'text': 10,
            'node_label': 6,
            'edge_label': 6,
        }
    
    # Ensure indices match if annotations are provided
    if row_annotations is not None: 
        row_annotations = row_annotations.loc[data.index]
    if col_annotations is not None:
        col_annotations = col_annotations.loc[data.columns]
    
    # Automatically enable colorbar for quantitative heatmaps
    if heatmap_type == 'quantitative' and not categorical_data:  
        show_colorbar = True
    
    # Determine colorbar orientation
    if colorbar_orientation is None:
        if colorbar_position in ['right', 'left']: 
            colorbar_orientation = 'vertical'
        else:
            colorbar_orientation = 'horizontal'
    
    # Create figure and axis if not provided
    if not use_existing_ax:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare colorbar keyword arguments
    if cbar_kws is None:  
        cbar_kws = {}
    
    # Determine how to handle colorbar
    colorbar_fig = None
    patch_colorbar_figs = {}
    
    # Variables for categorical data (need to be accessible later)
    categories = None
    cat_to_int = None
    custom_cmap = None
    
    # If using circle markers with significance, plot white background
    if significance_data is not None and significance_marker == 'circle':
        # Plot white background heatmap
        im = sns.heatmap(
            data,
            cmap=[circle_background],
            ax=ax,
            cbar=False,
            linewidths=heatmap_linewidths,
            linecolor=heatmap_linecolor,
            xticklabels=True,
            yticklabels=True,
            vmin=vmin,
            vmax=vmax,
            square=True if square else False,
            annot=False
        )
        cbar_enabled = False
    elif categorical_data:
        # Handle categorical data with custom palette
        # Create a mapping from categories to integers
        categories = sorted(categorical_palette.keys())
        cat_to_int = {cat: idx for idx, cat in enumerate(categories)}
        
        # Convert data to integer representation
        data_numeric = data.applymap(lambda x: cat_to_int.get(x, np.nan) if pd.notna(x) else np.nan)
        
        # Create custom colormap from palette
        colors = [categorical_palette[cat] for cat in categories]
        custom_cmap = ListedColormap(colors)
        
        # Plot heatmap
        im = sns.heatmap(
            data_numeric,
            cmap=custom_cmap,
            ax=ax,
            cbar=False,
            linewidths=heatmap_linewidths,
            linecolor=heatmap_linecolor,
            xticklabels=True,
            yticklabels=True,
            vmin=0,
            vmax=len(categories) - 1,
            square=True if square else False,
            annot=False
        )
        cbar_enabled = False
    else:
        # Normal heatmap plotting (existing code)
        if show_colorbar:
            if separate_colorbar:
                cbar_enabled = False
            elif colorbar_ax is not None:
                cbar_kws.update({'cax': colorbar_ax})
                cbar_enabled = True
            else:
                cbar_kws.update({
                    'orientation': colorbar_orientation,
                    'pad': colorbar_pad,
                    'label': colorbar_label if colorbar_label else '',
                    'location': colorbar_position
                })
                
                if 'fraction' not in cbar_kws:
                    if isinstance(colorbar_size, str) and '%' in colorbar_size:  
                        size_value = float(colorbar_size.rstrip('%')) / 100
                    else:  
                        size_value = 0.03
                    cbar_kws['fraction'] = size_value
                
                cbar_enabled = True
        else:
            cbar_enabled = False
        
        # Plot the heatmap using Seaborn
        im = sns.heatmap(
            data,
            cmap=cmap,
            ax=ax,
            cbar=cbar_enabled,
            cbar_kws=cbar_kws if cbar_enabled else None,
            linewidths=heatmap_linewidths,
            linecolor=heatmap_linecolor,
            xticklabels=True,
            yticklabels=True,
            vmin=vmin,
            vmax=vmax,
            center=center,
            robust=robust,
            square=True if square else False,
            annot=False
        )
    
    # IMPORTANT: Force tick positions and labels to be visible after seaborn plotting
    ax.set_xticks(np.arange(len(data.columns)) + 0.5)
    ax.set_yticks(np.arange(len(data.index)) + 0.5)
    
    # [Continue with colorbar handling - keeping existing code]
    # Handle colorbar customization or creation
    # For circle markers, create colorbar manually from data colormap
    if significance_data is not None and significance_marker == 'circle' and show_colorbar:
        if separate_colorbar:
            if colorbar_figsize is None:
                if colorbar_orientation == 'vertical':
                    colorbar_figsize = (2, 6)
                else:
                    colorbar_figsize = (6, 1.5)
            
            colorbar_fig = plt.figure(figsize=colorbar_figsize)
            cbar_ax_separate = colorbar_fig.add_axes([0.1, 0.1, 0.8, 0.8])
            
            # Create normalization and colorbar
            if center is not None:
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=vmin if vmin is not None else data.min().min(),
                                   vcenter=center,
                                   vmax=vmax if vmax is not None else data.max().max())
            else:
                norm = Normalize(vmin=vmin if vmin is not None else data.min().min(),
                               vmax=vmax if vmax is not None else data.max().max())
            
            cbar = plt.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax_separate,
                orientation=colorbar_orientation
            )
            
            if colorbar_label:
                cbar.set_label(colorbar_label,
                              fontsize=font_size.get('cbar_label', font_size['label']),
                              rotation=90 if colorbar_orientation == 'vertical' else 0,
                              labelpad=10)
            
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
            
            cbar.ax.tick_params(labelsize=font_size.get('cbar_ticks', font_size['ticks_label']))
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('black')
            
            if save_path is not None:
                base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, 'png')
                cbar_save_path = f"{base}_colorbar.{ext}"
                colorbar_fig.savefig(cbar_save_path, dpi=dpi, bbox_inches='tight')
        elif colorbar_ax is not None or colorbar_coords is not None:
            # Create colorbar in specified location
            if center is not None:
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=vmin if vmin is not None else data.min().min(),
                                   vcenter=center,
                                   vmax=vmax if vmax is not None else data.max().max())
            else:
                norm = Normalize(vmin=vmin if vmin is not None else data.min().min(),
                               vmax=vmax if vmax is not None else data.max().max())
            
            if colorbar_ax is not None:
                cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
            else:
                cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                                   orientation=colorbar_orientation, pad=colorbar_pad)
                if colorbar_coords is not None:
                    cbar.ax.set_position(colorbar_coords)
            
            if colorbar_label:
                cbar.set_label(colorbar_label,
                              fontsize=font_size.get('cbar_label', font_size['label']),
                              rotation=90 if colorbar_orientation == 'vertical' else 0,
                              labelpad=10)
            
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
            
            cbar.ax.tick_params(labelsize=font_size.get('cbar_ticks', font_size['ticks_label']))
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('black')
    
    elif show_colorbar and not categorical_data:
        # Handle colorbar for normal (non-circle) heatmaps
        if separate_colorbar:
            # Create separate colorbar figure
            if colorbar_figsize is None:
                if colorbar_orientation == 'vertical':
                    colorbar_figsize = (2, 6)
                else:
                    colorbar_figsize = (6, 1.5)
            
            colorbar_fig = plt.figure(figsize=colorbar_figsize)
            cbar_ax_separate = colorbar_fig.add_axes([0.1, 0.1, 0.8, 0.8])
            
            # Get normalization from the heatmap
            if center is not None:
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=vmin if vmin is not None else data.min().min(),
                                   vcenter=center,
                                   vmax=vmax if vmax is not None else data.max().max())
            else:
                norm = Normalize(vmin=vmin if vmin is not None else data.min().min(),
                               vmax=vmax if vmax is not None else data.max().max())
            
            cbar = plt.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax_separate,
                orientation=colorbar_orientation
            )
            
            if colorbar_label:
                cbar.set_label(colorbar_label,
                              fontsize=font_size.get('cbar_label', font_size['label']),
                              rotation=90 if colorbar_orientation == 'vertical' else 0,
                              labelpad=10)
            
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
            
            cbar.ax.tick_params(labelsize=font_size.get('cbar_ticks', font_size['ticks_label']))
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('black')
            
            if save_path is not None:
                base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, 'png')
                cbar_save_path = f"{base}_colorbar.{ext}"
                colorbar_fig.savefig(cbar_save_path, dpi=dpi, bbox_inches='tight')
        elif cbar_enabled and hasattr(im, 'collections') and len(im.collections) > 0:
            # Add colorbar to main plot (seaborn already created it)
            cbar = ax.collections[0].colorbar

            if colorbar_coords is not None and colorbar_ax is None:
                cbar.ax.set_position(colorbar_coords)
            
            if colorbar_label: 
                cbar.set_label(colorbar_label,
                              fontsize=font_size.get('cbar_label', font_size['label']),
                              rotation=90 if colorbar_orientation == 'vertical' else 0,
                              labelpad=10)
            
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
            
            cbar.ax.tick_params(labelsize=font_size.get('cbar_ticks', font_size['ticks_label']))
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('black')
    
    # Set tick label positions
    ax.xaxis.tick_top() if xticklabels_position == 'top' else ax.xaxis.tick_bottom()
    ax.yaxis.tick_right() if yticklabels_position == 'right' else ax.yaxis.tick_left()
    
    # Ensure ticks are visible and properly positioned
    ax.xaxis.set_ticks_position('top' if xticklabels_position == 'top' else 'bottom')
    ax.xaxis.set_label_position('top' if xticklabels_position == 'top' else 'bottom')
    ax.yaxis.set_ticks_position('right' if yticklabels_position == 'right' else 'left')
    ax.yaxis.set_label_position('right' if yticklabels_position == 'right' else 'left')
    
    # Customize tick labels with custom labels if provided
    ax.set_xticklabels(
        xticklabels,
        rotation=xticklabels_rotation,
        # Updated HA: Center if 0 or 90, otherwise keep existing logic
        ha='center' if xticklabels_rotation in [0, 90] else (
            ('left' if xticklabels_rotation < 0 else 'right') if xticklabels_position == 'bottom' 
            else ('right' if xticklabels_rotation < 0 else 'left')
        ),
        # Updated VA: Center if 0, otherwise Top/Bottom based on position
        va='center' if xticklabels_rotation == 0 else ('top' if xticklabels_position == 'bottom' else 'bottom'),
        fontsize=font_size['ticks_label']
    )

    ax.set_yticklabels(
        yticklabels,
        rotation=yticklabels_rotation,
        # Updated HA: Center if 90, otherwise Right/Left based on position
        ha='center' if yticklabels_rotation == 90 else ('right' if yticklabels_position == 'left' else 'left'),
        # Updated VA: Center if 0 or 90, otherwise keep existing logic
        va='center' if yticklabels_rotation in [0, 90] else (
            ('top' if yticklabels_rotation > 0 else 'bottom') if yticklabels_position == 'left'
            else ('bottom' if yticklabels_rotation > 0 else 'top')
        ),
        fontsize=font_size['ticks_label']
    )
    
    # Customize axis labels
    ax.set_xlabel(xlabel, fontsize=font_size['label'], labelpad=15)
    ax.set_ylabel(ylabel, fontsize=font_size['label'], labelpad=15)
    
    # Get the axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Get heatmap dimensions
    n_rows = len(data.index)
    n_cols = len(data.columns)
    heatmap_width = xlim[1] - xlim[0]
    heatmap_height = ylim[1] - ylim[0]
    
    # Calculate automatic patch dimensions for each annotation
    if row_annotation_col is not None:
        n_row_annots = len(row_annotation_col)
        if row_patch_width is None:
            row_patch_width = [None] * n_row_annots
        
        for idx in range(n_row_annots):
            if row_patch_auto_width and row_patch_width[idx] is None:
                calculated_row_patch_width = heatmap_width * patch_width_ratio
                row_patch_width[idx] = max(calculated_row_patch_width, 0.3)
            elif row_patch_width[idx] is None:
                row_patch_width[idx] = 0.5
    
    if col_annotation_col is not None:
        n_col_annots = len(col_annotation_col)
        if col_patch_height is None:
            col_patch_height = [None] * n_col_annots
        
        for idx in range(n_col_annots):
            if col_patch_auto_height and col_patch_height[idx] is None:
                calculated_col_patch_height = heatmap_height * patch_height_ratio
                col_patch_height[idx] = max(calculated_col_patch_height, 1.0)
            elif col_patch_height[idx] is None:
                col_patch_height[idx] = 2.0
    
    # Calculate total width/height for all patches including spacing
    total_row_patch_width = 0
    if row_annotation_col is not None and row_palette is not None:
        total_row_patch_width = sum(row_patch_width) + row_patch_spacing * (len(row_patch_width) - 1)
    
    total_col_patch_height = 0
    if col_annotation_col is not None and col_palette is not None:  
        total_col_patch_height = sum(col_patch_height) + col_patch_spacing * (len(col_patch_height) - 1)
    
    # Calculate automatic tick padding based on patch dimensions
    if auto_tick_padding:
        fig_dpi = fig.dpi
        
        if tick_pad_y is None and total_row_patch_width > 0:
            bbox = ax.get_position()
            ax_width_inches = bbox.width * calc_width if use_existing_ax else bbox.width * figsize[0]
            data_per_inch_x = heatmap_width / ax_width_inches
            patch_width_inches = total_row_patch_width / data_per_inch_x
            patch_width_points = patch_width_inches * 35
            tick_pad_y = base_tick_pad + (patch_width_points * tick_pad_ratio)
        elif tick_pad_y is None:  
            tick_pad_y = base_tick_pad
        
        if tick_pad_x is None and total_col_patch_height > 0:
            bbox = ax.get_position()
            ax_height_inches = bbox.height * calc_height if use_existing_ax else bbox.height * figsize[1]
            data_per_inch_y = heatmap_height / ax_height_inches
            patch_height_inches = total_col_patch_height / data_per_inch_y
            patch_height_points = patch_height_inches * 35
            tick_pad_x = base_tick_pad + (patch_height_points * tick_pad_ratio)
        elif tick_pad_x is None: 
            tick_pad_x = base_tick_pad
    else:
        if tick_pad_x is None:
            tick_pad_x = 20
        if tick_pad_y is None:
            tick_pad_y = 20
    
    ax.tick_params(axis='x', which='both', length=0, pad=tick_pad_x, 
                   top=(xticklabels_position == 'top'), 
                   bottom=(xticklabels_position == 'bottom'),
                   labeltop=(xticklabels_position == 'top'), 
                   labelbottom=(xticklabels_position == 'bottom'))
    ax.tick_params(axis='y', which='both', length=0, pad=tick_pad_y,
                   left=(yticklabels_position == 'left'), 
                   right=(yticklabels_position == 'right'),
                   labelleft=(yticklabels_position == 'left'), 
                   labelright=(yticklabels_position == 'right'))
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    legend_dict = {}
    
        # Add patches for rows - position based on yticklabels_position
    if row_annotations is not None and row_annotation_col is not None and row_palette is not None:
        for annot_idx in range(len(row_annotation_col)):
            if yticklabels_position == 'left':
                start_x = xlim[0] - total_row_patch_width + sum(row_patch_width[:annot_idx]) + row_patch_spacing * annot_idx
            else:
                start_x = xlim[1] + sum(row_patch_width[:annot_idx]) + row_patch_spacing * annot_idx
            
            col_name = row_annotation_col[annot_idx]
            col_data = row_annotations[col_name]
            
            # Check if data is quantitative or categorical
            is_quantitative = _is_numeric_column(col_data)
            
            # Check if palette is a colormap (string or Colormap object)
            is_colormap = isinstance(row_palette[annot_idx], (str, Colormap))
            
            if is_quantitative and is_colormap:
                # Quantitative (continuous) data - use colormap
                from matplotlib import cm
                from matplotlib.colors import TwoSlopeNorm
                
                # Get colormap
                if isinstance(row_palette[annot_idx], str):
                    cmap_row = cm.get_cmap(row_palette[annot_idx])
                else:
                    # Already a colormap object
                    cmap_row = row_palette[annot_idx]
                
                # Get vmin/vmax/center
                vmin_row = row_vmin[annot_idx] if (row_vmin and row_vmin[annot_idx] is not None) else col_data.min()
                vmax_row = row_vmax[annot_idx] if (row_vmax and row_vmax[annot_idx] is not None) else col_data.max()
                center_row = row_center[annot_idx] if (row_center and row_center[annot_idx] is not None) else None
                
                # Create normalization (centered or regular)
                if center_row is not None:
                    norm_row = TwoSlopeNorm(vmin=vmin_row, vcenter=center_row, vmax=vmax_row)
                else:
                    norm_row = Normalize(vmin=vmin_row, vmax=vmax_row)
                
                for i, row_idx in enumerate(data.index):
                    value = row_annotations.loc[row_idx, col_name]
                    if pd.notna(value):
                        # Map value to color
                        color = cmap_row(norm_row(value))
                    else:
                        color = 'lightgrey'
                    
                    rect = Rectangle(
                        (start_x - 0.2, i) if yticklabels_position == 'left' else (start_x + 0.2, i),
                        row_patch_width[annot_idx], 1,
                        alpha=row_patch_alpha,
                        linewidth=patch_linewidths,
                        edgecolor=patch_linecolor,
                        facecolor=color, clip_on=False, zorder=-1
                    )
                    ax.add_patch(rect)
                
                # Handle colorbar for this row annotation
                sep_cbar = row_separate_colorbar[annot_idx] if row_separate_colorbar else False
                
                if sep_cbar:
                    # Create separate colorbar
                    cbar_orient = row_colorbar_orientation[annot_idx] if row_colorbar_orientation else 'vertical'
                    cbar_figsize = row_colorbar_figsize[annot_idx] if row_colorbar_figsize else (
                        (2, 4) if cbar_orient == 'vertical' else (4, 1.5)
                    )
                    
                    cbar_fig = plt.figure(figsize=cbar_figsize)
                    cbar_ax_sep = cbar_fig.add_axes([0.1, 0.1, 0.8, 0.8])
                    
                    cbar_obj = plt.colorbar(
                        ScalarMappable(norm=norm_row, cmap=cmap_row),
                        cax=cbar_ax_sep,
                        orientation=cbar_orient
                    )
                    
                    # Apply colorbar styling
                    cbar_label_text = row_colorbar_label[annot_idx] if row_colorbar_label else None
                    if cbar_label_text:
                        cbar_obj.set_label(cbar_label_text,
                                          fontsize=font_size.get('cbar_label', 12),
                                          rotation=90 if cbar_orient == 'vertical' else 0,
                                          labelpad=10)
                    
                    if row_cbar_ticks and row_cbar_ticks[annot_idx] is not None:
                        cbar_obj.set_ticks(row_cbar_ticks[annot_idx])
                    
                    cbar_obj.ax.tick_params(labelsize=font_size.get('cbar_ticks', 10))
                    cbar_obj.outline.set_linewidth(0.5)
                    cbar_obj.outline.set_edgecolor('black')
                    
                    patch_colorbar_figs[f'row_{annot_idx}'] = cbar_fig
                    
                    if save_path is not None:
                        base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, 'png')
                        cbar_save_path = f"{base}_row_{annot_idx}_colorbar.{ext}"
                        cbar_fig.savefig(cbar_save_path, dpi=dpi, bbox_inches='tight')
                
                # Create legend entry for quantitative data
                legend_dict[f'row_{annot_idx}'] = {
                    'handles': [],
                    'title': row_legend_title[annot_idx],
                    'n_items': 0,
                    'type': 'quantitative',
                    'cmap': cmap_row,
                    'norm': norm_row,
                    'label': row_legend_title[annot_idx],
                    'separate_colorbar': sep_cbar
                }
            
            else:
                # Categorical data - use palette dict
                present_categories = set()
                for i, row_idx in enumerate(data.index):
                    category = row_annotations.loc[row_idx, col_name]
                    present_categories.add(category)
                    color = row_palette[annot_idx].get(category, 'grey')
                    rect = Rectangle(
                        (start_x - row_patch_offset, i) if yticklabels_position == 'left' else (start_x + row_patch_offset, i),
                        row_patch_width[annot_idx], 1,
                        alpha=row_patch_alpha,
                        linewidth=patch_linewidths,
                        edgecolor=patch_linecolor,
                        facecolor=color, clip_on=False, zorder=-1
                    )
                    ax.add_patch(rect)
                
                legend_elements_row = [
                    Rectangle((0, 0), 1, 1, fc=color,
                             ec=legend_patch_linecolor,
                             linewidth=legend_patch_linewidths, label=category)
                    for category, color in row_palette[annot_idx].items()
                    if category in present_categories
                ]
                legend_dict[f'row_{annot_idx}'] = {
                    'handles': legend_elements_row,
                    'title': row_legend_title[annot_idx],
                    'n_items': len(legend_elements_row),
                    'type': 'categorical'
                }
    
    # Add patches for columns - position based on xticklabels_position
    if col_annotations is not None and col_annotation_col is not None and col_palette is not None:
        for annot_idx in range(len(col_annotation_col)):
            if xticklabels_position == 'bottom':
                start_y = ylim[0] + sum(col_patch_height[:annot_idx]) + col_patch_spacing * annot_idx
            else:
                start_y = ylim[1] - total_col_patch_height + sum(col_patch_height[:annot_idx]) + col_patch_spacing * annot_idx
            
            col_name = col_annotation_col[annot_idx]
            col_data = col_annotations[col_name]
            
            # Check if data is quantitative or categorical
            is_quantitative = _is_numeric_column(col_data)
            
            # Check if palette is a colormap (string or Colormap object)
            is_colormap = isinstance(col_palette[annot_idx], (str, Colormap))
            
            if is_quantitative and is_colormap:
                # Quantitative (continuous) data - use colormap
                from matplotlib import cm
                from matplotlib.colors import TwoSlopeNorm
                
                # Get colormap
                if isinstance(col_palette[annot_idx], str):
                    cmap_col = cm.get_cmap(col_palette[annot_idx])
                else:
                    # Already a colormap object
                    cmap_col = col_palette[annot_idx]
                
                # Get vmin/vmax/center
                vmin_col = col_vmin[annot_idx] if (col_vmin and col_vmin[annot_idx] is not None) else col_data.min()
                vmax_col = col_vmax[annot_idx] if (col_vmax and col_vmax[annot_idx] is not None) else col_data.max()
                center_col = col_center[annot_idx] if (col_center and col_center[annot_idx] is not None) else None
                
                # Create normalization (centered or regular)
                if center_col is not None:
                    norm_col = TwoSlopeNorm(vmin=vmin_col, vcenter=center_col, vmax=vmax_col)
                else:
                    norm_col = Normalize(vmin=vmin_col, vmax=vmax_col)
                
                for j, col_idx in enumerate(data.columns):
                    value = col_annotations.loc[col_idx, col_name]
                    if pd.notna(value):
                        # Map value to color
                        color = cmap_col(norm_col(value))
                    else:
                        color = 'lightgrey'
                    
                    rect = Rectangle(
                        (j, start_y + col_patch_offset) if xticklabels_position == 'bottom' else (j, start_y - col_patch_offset),
                        1, col_patch_height[annot_idx],
                        alpha=col_patch_alpha,
                        linewidth=patch_linewidths,
                        edgecolor=patch_linecolor,
                        facecolor=color, clip_on=False, zorder=-1
                    )
                    ax.add_patch(rect)
                
                # Handle colorbar for this col annotation
                sep_cbar = col_separate_colorbar[annot_idx] if col_separate_colorbar else False
                
                if sep_cbar:
                    # Create separate colorbar
                    cbar_orient = col_colorbar_orientation[annot_idx] if col_colorbar_orientation else 'vertical'
                    cbar_figsize = col_colorbar_figsize[annot_idx] if col_colorbar_figsize else (
                        (2, 4) if cbar_orient == 'vertical' else (4, 1.5)
                    )
                    
                    cbar_fig = plt.figure(figsize=cbar_figsize)
                    cbar_ax_sep = cbar_fig.add_axes([0.1, 0.1, 0.8, 0.8])
                    
                    cbar_obj = plt.colorbar(
                        ScalarMappable(norm=norm_col, cmap=cmap_col),
                        cax=cbar_ax_sep,
                        orientation=cbar_orient
                    )
                    
                    # Apply colorbar styling
                    cbar_label_text = col_colorbar_label[annot_idx] if col_colorbar_label else None
                    if cbar_label_text:
                        cbar_obj.set_label(cbar_label_text,
                                          fontsize=font_size.get('cbar_label', 12),
                                          rotation=90 if cbar_orient == 'vertical' else 0,
                                          labelpad=10)
                    
                    if col_cbar_ticks and col_cbar_ticks[annot_idx] is not None:
                        cbar_obj.set_ticks(col_cbar_ticks[annot_idx])
                    
                    cbar_obj.ax.tick_params(labelsize=font_size.get('cbar_ticks', 10))
                    cbar_obj.outline.set_linewidth(0.5)
                    cbar_obj.outline.set_edgecolor('black')
                    
                    patch_colorbar_figs[f'col_{annot_idx}'] = cbar_fig
                    
                    if save_path is not None:
                        base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, 'png')
                        cbar_save_path = f"{base}_col_{annot_idx}_colorbar.{ext}"
                        cbar_fig.savefig(cbar_save_path, dpi=dpi, bbox_inches='tight')
                
                # Create legend entry for quantitative data
                legend_dict[f'col_{annot_idx}'] = {
                    'handles': [],
                    'title': col_legend_title[annot_idx],
                    'n_items': 0,
                    'type': 'quantitative',
                    'cmap': cmap_col,
                    'norm': norm_col,
                    'label': col_legend_title[annot_idx],
                    'separate_colorbar': sep_cbar
                }
            
            else:
                # Categorical data - use palette dict
                present_categories = set()
                for j, col_idx in enumerate(data.columns):
                    category = col_annotations.loc[col_idx, col_name]
                    present_categories.add(category)
                    color = col_palette[annot_idx].get(category, 'grey')
                    rect = Rectangle(
                        (j, start_y + 0.1) if xticklabels_position == 'bottom' else (j, start_y - 0.1),
                        1, col_patch_height[annot_idx],
                        alpha=col_patch_alpha,
                        linewidth=patch_linewidths,
                        edgecolor=patch_linecolor,
                        facecolor=color, clip_on=False, zorder=-1
                    )
                    ax.add_patch(rect)
                
                legend_elements_col = [
                    Rectangle((0, 0), 1, 1, fc=color,
                             ec=legend_patch_linecolor,
                             linewidth=legend_patch_linewidths, label=category)
                    for category, color in col_palette[annot_idx].items()
                    if category in present_categories
                ]
                legend_dict[f'col_{annot_idx}'] = {
                    'handles': legend_elements_col,
                    'title': col_legend_title[annot_idx],
                    'n_items': len(legend_elements_col),
                    'type': 'categorical'
                }
    
    # Add separation lines
    if row_separation_col is not None and row_annotations is not None:
        for sep_idx, sep_col in enumerate(row_separation_col):
            row_categories = row_annotations[sep_col].values
            for i in range(1, len(row_categories)):
                if row_categories[i] != row_categories[i-1]:
                    ax.axhline(
                        y=i, xmin=0, xmax=1,
                        color=row_separation_color[sep_idx],
                        linewidth=row_separation_linewidth[sep_idx],
                        linestyle=row_separation_linestyle[sep_idx],
                        alpha=row_separation_alpha[sep_idx],
                        clip_on=False, zorder=10 + sep_idx
                    )
    
    if col_separation_col is not None and col_annotations is not None: 
        for sep_idx, sep_col in enumerate(col_separation_col):
            col_categories = col_annotations[sep_col].values
            for j in range(1, len(col_categories)):
                if col_categories[j] != col_categories[j-1]:
                    ax.axvline(
                        x=j, ymin=0, ymax=1,
                        color=col_separation_color[sep_idx],
                        linewidth=col_separation_linewidth[sep_idx],
                        linestyle=col_separation_linestyle[sep_idx],
                        alpha=col_separation_alpha[sep_idx],
                        clip_on=False, zorder=10 + sep_idx
                    )
    
    # Add significance markers if provided
    if significance_data is not None:
        # Default significance function
        if significance_func is None:
            def default_pval_to_symbol(p):
                if p <= 1e-4:
                    return '****'
                elif p <= 1e-3:
                    return '***'
                elif p <= 1e-2:
                    return '**'
                elif p <= 5e-2:
                    return '*'
                else:
                    return 'ns'
            significance_func = default_pval_to_symbol
        
        # Default size mapping
        if significance_size_map is None:
            significance_size_map = {
                '****': 300,
                '***': 225,
                '**': 150,
                '*': 80,
                'ns': 30
            }
        
        # Calculate text size if not provided
        if significance_text_size is None:
            significance_text_size = font_size.get('annotation', 9)
        
        if significance_marker == 'circle':
            # Create colormap normalizer for data values
            if center is not None:
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=vmin if vmin is not None else data.min().min(),
                                   vcenter=center,
                                   vmax=vmax if vmax is not None else data.max().max())
            else:
                norm = Normalize(vmin=vmin if vmin is not None else data.min().min(),
                               vmax=vmax if vmax is not None else data.max().max())
            
            # Get colormap
            if isinstance(cmap, str):
                from matplotlib import cm
                colormap = cm.get_cmap(cmap)
            else:
                colormap = cmap
            
            # Collect data for all circles
            x_coords = []
            y_coords = []
            sizes = []
            colors = []
            
            # Apply significance markers
            for i, row_idx in enumerate(data.index):
                for j, col_idx in enumerate(data.columns):
                    pval = significance_data.loc[row_idx, col_idx]
                    value = data.loc[row_idx, col_idx]
                    
                    # Skip if either is NaN
                    if pd.isna(pval) or pd.isna(value):
                        continue
                    
                    # Get significance symbol and size
                    symbol = significance_func(pval)
                    marker_size = significance_size_map.get(symbol, 20)
                    
                    # Cell center coordinates
                    x_center = j + 0.5
                    y_center = i + 0.5
                    
                    # Get color from data value
                    color = colormap(norm(value))
                    
                    x_coords.append(x_center)
                    y_coords.append(y_center)
                    sizes.append(marker_size)
                    colors.append(color)
            
            # Plot all circles at once
            if len(x_coords) > 0:
                # Determine edge colors
                if significance_linewidth > 0:
                    if significance_edgecolor is not None:
                        edge_colors = significance_edgecolor
                    else:
                        edge_colors = colors
                else:
                    edge_colors = 'none'
                
                ax.scatter(
                    x_coords, y_coords,
                    s=sizes,
                    c=colors,
                    alpha=significance_alpha,
                    edgecolors=edge_colors,
                    linewidths=significance_linewidth,
                    zorder=10
                )
        
        else:
            # Original code for other marker types
            for i, row_idx in enumerate(data.index):
                for j, col_idx in enumerate(data.columns):
                    pval = significance_data.loc[row_idx, col_idx]
                    
                    if pd.isna(pval):
                        continue
                    
                    symbol = significance_func(pval)
                    marker_size = significance_size_map.get(symbol, 0)
                    
                    if marker_size > 0:
                        x_center = j + 0.5
                        y_center = i + 0.5
                        
                        if significance_marker == 'star':
                            ax.scatter(
                                x_center, y_center,
                                marker='*',
                                s=marker_size,
                                color=significance_color,
                                alpha=significance_alpha,
                                linewidths=significance_linewidth,
                                edgecolors=significance_color,
                                zorder=10
                            )
                        
                        elif significance_marker in ['asterisk', 'text']:
                            display_symbol = symbol if significance_marker == 'text' else symbol.replace('ns', '')
                            if display_symbol:
                                ax.text(
                                    x_center, y_center,
                                    display_symbol,
                                    ha='center', va='center',
                                    fontsize=significance_text_size,
                                    color=significance_color,
                                    alpha=significance_alpha,
                                    weight='bold',
                                    zorder=10
                                )
        
        # Add significance legend
        if show_significance_legend:
            from matplotlib.lines import Line2D
            sig_handles = []
            for symbol in sorted(significance_size_map.keys(), 
                               key=lambda x: significance_size_map[x], reverse=True):
                size = significance_size_map[symbol]
                if size > 0:
                    if significance_marker == 'circle':
                        handle = Line2D(
                            [0], [0],
                            marker='o',
                            color='w',
                            markerfacecolor='gray',
                            markeredgecolor='none',
                            markersize=np.sqrt(size)/2.5,
                            label=symbol,
                            linestyle='None'
                        )
                    elif significance_marker == 'star':
                        handle = Line2D(
                            [0], [0],
                            marker='*',
                            color='w',
                            markerfacecolor=significance_color,
                            markeredgecolor=significance_color,
                            markersize=np.sqrt(size)/2.5,
                            label=symbol,
                            linestyle='None'
                        )
                    else:
                        handle = Line2D(
                            [0], [0],
                            marker=f'${symbol}$',
                            color='w',
                            markerfacecolor=significance_color,
                            markersize=10,
                            label=symbol,
                            linestyle='None'
                        )
                    sig_handles.append(handle)
            
            if sig_handles:
                legend_dict['significance'] = {
                    'handles': sig_handles,
                    'title': significance_legend_title,
                    'n_items': len(sig_handles),
                    'type': 'categorical'
                }
    
    # Add cell annotations if requested
    if show_cell_annotations:
        # Determine what to display in cells
        if cell_annotation_data is not None:
            # Use provided annotation data
            annot_data = cell_annotation_data
        else:
            # Use original data values
            annot_data = data
        
        # Get colormap for auto text color
        if cell_annotation_color == 'auto':
            if categorical_data:
                # Use the custom categorical colormap
                color_norm = Normalize(vmin=0, vmax=len(categories) - 1)
                colormap_for_text = custom_cmap
            elif significance_data is not None and significance_marker == 'circle':
                # White background
                def get_cell_color(i, j):
                    return [1, 1, 1, 1]  # White
            else:
                # Use main heatmap colormap
                if center is not None:
                    from matplotlib.colors import TwoSlopeNorm
                    color_norm = TwoSlopeNorm(
                        vmin=vmin if vmin is not None else data.min().min(),
                        vcenter=center,
                        vmax=vmax if vmax is not None else data.max().max()
                    )
                else:
                    color_norm = Normalize(
                        vmin=vmin if vmin is not None else data.min().min(),
                        vmax=vmax if vmax is not None else data.max().max()
                    )
                
                if isinstance(cmap, str):
                    from matplotlib import cm
                    colormap_for_text = cm.get_cmap(cmap)
                else:
                    colormap_for_text = cmap
        
        # Add text annotations
        for i, row_idx in enumerate(data.index):
            for j, col_idx in enumerate(data.columns):
                annot_value = annot_data.loc[row_idx, col_idx]
                
                if pd.isna(annot_value):
                    continue
                
                # Format the annotation text
                if isinstance(annot_value, (int, float)):
                    # Numeric value - apply format
                    text = cell_annotation_format.format(x=annot_value)
                else:
                    # String/categorical value - use as is
                    text = str(annot_value)
                
                # Determine text color based on DATA (not annotation)
                if cell_annotation_color == 'auto':
                    if categorical_data:
                        # Get background color from data category
                        data_category = data.loc[row_idx, col_idx]
                        if pd.notna(data_category):
                            cat_idx = cat_to_int.get(data_category, 0)
                            bg_color = colormap_for_text(color_norm(cat_idx))
                        else:
                            bg_color = [0.8, 0.8, 0.8, 1]  # Light gray for NaN
                    elif significance_data is not None and significance_marker == 'circle':
                        bg_color = [1, 1, 1, 1]  # White
                    else:
                        # Get background color from data value
                        data_value = data.loc[row_idx, col_idx]
                        if pd.notna(data_value):
                            bg_color = colormap_for_text(color_norm(data_value))
                        else:
                            bg_color = [0.8, 0.8, 0.8, 1]  # Light gray
                    
                    text_color = _get_text_color_for_background(bg_color, cell_annotation_threshold)
                else:
                    text_color = cell_annotation_color
                
                # Add text
                x_center = j + 0.5
                y_center = i + 0.5
                
                ax.text(
                    x_center, y_center,
                    text,
                    ha='center', va='center',
                    fontsize=font_size['annotation'],
                    color=text_color,
                    weight=cell_annotation_weight,
                    zorder=15  # Above significance markers
                )
    
    # Create legend for categorical heatmap data
    if categorical_data:
        # Determine legend labels
        if categorical_legend_labels is not None:
            # Use custom labels
            legend_handles = [
                Rectangle((0, 0), 1, 1, fc=categorical_palette[cat], 
                         ec=legend_patch_linecolor,
                         linewidth=legend_patch_linewidths,
                         label=categorical_legend_labels[cat])
                for cat in categories
            ]
        else:
            # Use category values as labels
            legend_handles = [
                Rectangle((0, 0), 1, 1, fc=categorical_palette[cat], 
                         ec=legend_patch_linecolor,
                         linewidth=legend_patch_linewidths,
                         label=cat)
                for cat in categories
            ]
        
        legend_dict['categorical'] = {
            'handles': legend_handles,
            'title': value_legend_title,
            'n_items': len(categories),
            'type': 'categorical'
        }
    
    # Handle legends
    if legend_order is None:
        legend_order = []
        if row_annotation_col is not None:
            legend_order.extend([f'row_{i}' for i in range(len(row_annotation_col))])
        if col_annotation_col is not None:
            legend_order.extend([f'col_{i}' for i in range(len(col_annotation_col))])
        if categorical_data:
            legend_order.append('categorical')
        if significance_data is not None and show_significance_legend:
            legend_order.append('significance')
        if not categorical_data:
            legend_order.append('value')
    
    legends_to_plot = [key for key in legend_order if key in legend_dict]
    legend_figs = {}

    def get_ncol(n_items, orientation):
        return n_items if orientation == 'horizontal' else 1

    if separate_legends:
        for key in legends_to_plot:
            legend_info = legend_dict[key]
            
            # Check if it's a quantitative legend that already has separate colorbar
            if legend_info.get('type') == 'quantitative':
                if legend_info.get('separate_colorbar', False):
                    # Already created as separate colorbar, skip
                    continue
                else:
                    # Create a colorbar-style legend
                    cbar_orient = colorbar_orientation if colorbar_orientation else 'vertical'
                    figsize_legend = (2, 4) if cbar_orient == 'vertical' else (4, 1.5)
                    
                    l_fig = plt.figure(figsize=figsize_legend)
                    l_ax = l_fig.add_axes([0.1, 0.1, 0.8, 0.8])
                    
                    cbar = plt.colorbar(
                        ScalarMappable(norm=legend_info['norm'], cmap=legend_info['cmap']),
                        cax=l_ax,
                        orientation=cbar_orient
                    )
                    cbar.set_label(legend_info['label'], fontsize=font_size.get('legend_title', 12))
                    cbar.ax.tick_params(labelsize=font_size.get('legend', 10))
                    cbar.outline.set_linewidth(0.5)
                    cbar.outline.set_edgecolor('black')
                    
                    legend_figs[key] = l_fig
            else:
                # Categorical legend (existing code)
                n_items = legend_info['n_items']
                if legend_orientation == 'horizontal':
                    figsize_legend = (n_items * 1.5, 1)
                else:
                    figsize_legend = (2, n_items * 0.5 + 0.5)
                
                l_fig = plt.figure(figsize=figsize_legend)
                l_ax = l_fig.add_subplot(111)
                l_ax.axis('off')
                
                l_ax.legend(
                    handles=legend_info['handles'],
                    title=legend_info['title'],
                    loc='center',
                    ncol=get_ncol(n_items, legend_orientation),
                    frameon=False,
                    fontsize=font_size.get('legend', 10),
                    title_fontsize=font_size.get('legend_title', 12),
                    alignment='left'
                )
                legend_figs[key] = l_fig
            
            if save_path is not None and key in legend_figs:
                base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, 'png')
                l_save_path = f"{base}_legend_{key}.{ext}"
                legend_figs[key].savefig(l_save_path, dpi=dpi, bbox_inches='tight')
    else:
        if legend_auto_spacing and legends_to_plot:
            legend_heights = []
            for key in legends_to_plot:  
                legend_info = legend_dict[key]
                if legend_info.get('type') == 'quantitative' and legend_info.get('separate_colorbar', False):
                    continue
                n_items = legend_info.get('n_items', 0)
                if legend_info.get('type') == 'quantitative':
                    estimated_height = 0.15
                elif legend_orientation == 'horizontal':
                    estimated_height = 0.06 
                else:
                    estimated_height = 0.04 + (n_items * 0.03)
                legend_heights.append(estimated_height)
            
            total_legend_height = sum(legend_heights) + (len(legend_heights) - 1) * legend_spacing
            
            if legend_alignment == 'top':
                start_y = 0.95
            elif legend_alignment == 'center':
                start_y = 0.5 + (total_legend_height / 2)
            elif legend_alignment == 'bottom':
                start_y = 0.05 + total_legend_height
            else: 
                start_y = 0.95 

            legend_positions = []
            current_y = start_y
            for height in legend_heights:
                legend_positions.append(current_y - height / 2)
                current_y -= (height + legend_spacing)
        else:
            if legend_alignment == 'top': start_y = 0.95
            elif legend_alignment == 'center': start_y = 0.5
            elif legend_alignment == 'bottom': start_y = 0.05 + (len(legends_to_plot) - 1) * legend_spacing
            else: start_y = 0.95
            legend_positions = [start_y - i * legend_spacing for i in range(len(legends_to_plot))]

        if show_colorbar and colorbar_position == legend_position and not separate_colorbar:
            if legend_position == 'right': legend_bbox_x = legend_bbox_x + 0.15
            elif legend_position == 'left': legend_bbox_x = legend_bbox_x - 0.15

        if legend_position == 'right':  
            legend_loc = 'center left'
            bbox_x = legend_bbox_x
        elif legend_position == 'left':
            legend_loc = 'center right'
            bbox_x = -0.02
        elif legend_position == 'top':
            legend_loc = 'lower center'
            bbox_x = 0.5
        elif legend_position == 'bottom':
            legend_loc = 'upper center'
            bbox_x = 0.5
        else:
            legend_loc = 'center left'
            bbox_x = legend_bbox_x

        pos_idx = 0
        for key in legends_to_plot:
            legend_info = legend_dict[key]
            
            # Skip quantitative legends with separate colorbar in non-separate mode
            if legend_info.get('type') == 'quantitative':
                if legend_info.get('separate_colorbar', False):
                    continue
                else:
                    pos_idx += 1
                    continue
            
            y_position = legend_positions[pos_idx]
            pos_idx += 1
            
            fig.legend(
                handles=legend_info['handles'],
                title=legend_info['title'],
                loc=legend_loc,
                bbox_to_anchor=(bbox_x, y_position),
                ncol=get_ncol(legend_info['n_items'], legend_orientation),
                frameon=True,
                fontsize=font_size.get('legend', 10),
                title_fontsize=font_size.get('legend_title', 12),
                alignment='left'
            )

    if title:
        ax.set_title(title, fontsize=font_size.get('title', 14), pad=50)
    
    if not use_existing_ax:
        plt.tight_layout()

    if save_path is not None and not use_existing_ax:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=False)

    return fig, ax, legend_figs, colorbar_fig, patch_colorbar_figs
    


# ------------------------------------------------------------------------
# Annotated Heatmap with Row/Column Categories and Dendrograms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap, Colormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from typing import Optional, Dict, Tuple, Union, Literal, List, Sequence, Callable
from scipy.cluster import hierarchy as sch


def plot_annotated_clustermap(
    data: pd.DataFrame,
    transpose: bool = False,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    ax_size: Optional[Tuple[float, float]] = None,
    row_annotations: Optional[pd.DataFrame] = None,
    col_annotations: Optional[pd.DataFrame] = None,
    row_palette: Optional[Union[Dict[str, str], List[Union[Dict[str, str], str, Colormap]]]] = None,
    col_palette: Optional[Union[Dict[str, str], List[Union[Dict[str, str], str, Colormap]]]] = None,
    row_vmin: Optional[Union[float, List[float]]] = None,
    row_vmax: Optional[Union[float, List[float]]] = None,
    row_center: Optional[Union[float, List[float]]] = None,
    col_vmin: Optional[Union[float, List[float]]] = None,
    col_vmax: Optional[Union[float, List[float]]] = None,
    col_center: Optional[Union[float, List[float]]] = None,
    row_colorbar_position: Optional[Union[str, List[str]]] = None,
    row_colorbar_size: Optional[Union[str, List[str]]] = None,
    row_colorbar_pad: Optional[Union[float, List[float]]] = None,
    row_colorbar_label: Optional[Union[str, List[str]]] = None,
    row_cbar_ticks: Optional[Union[Sequence[float], List[Sequence[float]]]] = None,
    row_colorbar_orientation: Optional[Union[str, List[str]]] = None,
    row_colorbar_coords: Optional[Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]] = None,
    row_colorbar_ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
    row_separate_colorbar: Optional[Union[bool, List[bool]]] = None,
    row_colorbar_figsize: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    col_colorbar_position: Optional[Union[str, List[str]]] = None,
    col_colorbar_size: Optional[Union[str, List[str]]] = None,
    col_colorbar_pad: Optional[Union[float, List[float]]] = None,
    col_colorbar_label: Optional[Union[str, List[str]]] = None,
    col_cbar_ticks: Optional[Union[Sequence[float], List[Sequence[float]]]] = None,
    col_colorbar_orientation: Optional[Union[str, List[str]]] = None,
    col_colorbar_coords: Optional[Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]] = None,
    col_colorbar_ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
    col_separate_colorbar: Optional[Union[bool, List[bool]]] = None,
    col_colorbar_figsize: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    figsize: Tuple[float, float] = (10, 14),
    square: bool = False,
    cmap: Union[str, LinearSegmentedColormap] = 'Blues',
    font_scale: float = 13,
    title: str = 'Annotated Clustermap',
    xlabel: str = 'Columns',
    ylabel: str = 'Rows',
    row_patch_width: Optional[Union[float, List[float]]] = None,
    col_patch_height: Optional[Union[float, List[float]]] = None,
    row_patch_auto_width: bool = True,
    col_patch_auto_height: bool = True,
    patch_width_ratio: float = 0.05,
    patch_height_ratio: float = 0.05,
    row_annotation_col: Optional[Union[str, List[str]]] = None,
    col_annotation_col: Optional[Union[str, List[str]]] = None,
    row_legend_title: Union[str, List[str]] = 'Row Categories',
    col_legend_title: Union[str, List[str]] = 'Column Categories',
    value_legend_title: str = 'Values',
    value_legend_labels: Optional[Dict] = None,
    legend_position: Literal['right', 'left', 'top', 'bottom'] = 'right',
    legend_alignment: Literal['top', 'center', 'bottom'] = 'top',
    legend_bbox_x: float = 1.02,
    legend_auto_spacing: bool = True,
    legend_spacing: float = 0.08,
    legend_order: Optional[list] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    show_colorbar: bool = False,
    colorbar_position: Literal['right', 'left', 'top', 'bottom'] = 'right',
    colorbar_size: str = '5%',
    colorbar_pad: float = 0.05,
    colorbar_label: Optional[str] = None,
    cbar_ticks: Optional[Sequence[float]] = None,
    colorbar_orientation: Optional[Literal['vertical', 'horizontal']] = None,
    colorbar_coords: Optional[Tuple[float, float, float, float]] = None,
    colorbar_ax: Optional[plt.Axes] = None,
    separate_colorbar: bool = False,
    colorbar_figsize: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    robust: bool = False,
    heatmap_type: Literal['qualitative', 'quantitative'] = 'qualitative',
    heatmap_linewidths: float = 0.5,
    heatmap_linecolor: str = 'grey',
    patch_linewidths: float = 0.0,
    patch_linecolor: str = 'grey',
    legend_patch_linewidths: Optional[float] = 0.5,
    legend_patch_linecolor: Optional[str] = 'gray',
    row_patch_alpha: float = 1.0,
    col_patch_alpha: float = 1.0,
    xticklabels: Optional[Union[pd.Series, List, np.ndarray]] = None,
    yticklabels: Optional[Union[pd.Series, List, np.ndarray]] = None,
    xticklabels_rotation: float = 45,
    yticklabels_rotation: float = 0,
    xticklabels_position: Literal['top', 'bottom'] = 'bottom',
    yticklabels_position: Literal['left', 'right'] = 'left',
    auto_tick_padding: bool = True,
    tick_pad_x: Optional[float] = None,
    tick_pad_y: Optional[float] = None,
    tick_pad_ratio: float = 1.5,
    base_tick_pad: float = 5.0,
    font_size_func: Optional[Callable] = None,
    cbar_kws: Optional[Dict] = None,
    row_patch_spacing: float = 0.0,
    col_patch_spacing: float = 0.0,
    row_separation_col: Optional[Union[str, List[str]]] = None,
    col_separation_col: Optional[Union[str, List[str]]] = None,
    row_separation_linewidth: Union[float, List[float]] = 2.0,
    col_separation_linewidth: Union[float, List[float]] = 2.0,
    row_separation_color: Union[str, List[str]] = 'black',
    col_separation_color: Union[str, List[str]] = 'black',
    row_separation_linestyle: Union[str, List[str]] = '-',
    col_separation_linestyle: Union[str, List[str]] = '-',
    row_separation_alpha: Union[float, List[float]] = 1.0,
    col_separation_alpha: Union[float, List[float]] = 1.0,
    separate_legends: bool = False,
    legend_orientation: Literal['vertical', 'horizontal'] = 'vertical',
    # Significance overlay parameters
    significance_data: Optional[pd.DataFrame] = None,
    significance_func: Optional[Callable] = None,
    significance_marker: Literal['circle', 'star', 'asterisk', 'text'] = 'circle',
    significance_size_map: Optional[Dict[str, float]] = None,
    significance_color: str = 'black',
    significance_alpha: float = 1.0,
    significance_linewidth: float = 0.5,
    significance_edgecolor: Optional[str] = None,
    significance_text_size: Optional[float] = None,
    show_significance_legend: bool = True,
    significance_legend_title: str = 'Significance',
    circle_background: str = 'white',
    # Dendrogram-specific parameters
    row_linkage: Optional[np.ndarray] = None,
    col_linkage: Optional[np.ndarray] = None,
    row_dendrogram_ax: Optional[plt.Axes] = None,
    col_dendrogram_ax: Optional[plt.Axes] = None,
    dendrogram_ratio: float = 0.2,
    row_cluster: bool = False,
    col_cluster: bool = False,
    dendrogram_kws: Optional[Dict] = None,
    row_dendrogram_kws: Optional[Dict] = None,
    col_dendrogram_kws: Optional[Dict] = None,
    row_dendrogram_color: Optional[Union[str, Literal['cluster']]] = 'cluster',
    col_dendrogram_color: Optional[Union[str, Literal['cluster']]] = 'cluster',
    row_color_threshold: Optional[float] = None,
    col_color_threshold: Optional[float] = None,
    row_cluster_colors: Optional[List[str]] = None,
    col_cluster_colors: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict, Optional[plt.Figure], Dict, Dict]:
    """
    Create an annotated heatmap with dendrograms for hierarchical clustering.
    This is a wrapper around plot_annotated_heatmap that adds clustering functionality.
    
    Parameters
    ----------
    data : pd.DataFrame
        The main data to plot as a heatmap.
    transpose : bool, default False
        If True, transpose the data matrix.
    [... all previous parameters from plot_annotated_heatmap ...]
    
    row_linkage : np.ndarray, optional
        Precomputed linkage matrix for rows.
    col_linkage : np.ndarray, optional
        Precomputed linkage matrix for columns.
    row_dendrogram_ax : plt.Axes, optional
        Axes to draw row dendrogram.
    col_dendrogram_ax : plt.Axes, optional
        Axes to draw column dendrogram.
    dendrogram_ratio : float, default 0.2
        Ratio of dendrogram size to heatmap size.
    row_cluster : bool, default False
        If True, cluster rows.
    col_cluster : bool, default False
        If True, cluster columns.
    dendrogram_kws : dict, optional
        Keyword arguments for both dendrograms.
    row_dendrogram_kws : dict, optional
        Keyword arguments for row dendrogram (overrides dendrogram_kws).
    col_dendrogram_kws : dict, optional
        Keyword arguments for column dendrogram (overrides dendrogram_kws).
    row_dendrogram_color : str or 'cluster', default 'cluster'
        Color scheme for row dendrogram.
    col_dendrogram_color : str or 'cluster', default 'cluster'
        Color scheme for column dendrogram.
    row_color_threshold : float, optional
        Distance threshold for coloring row dendrogram clusters.
    col_color_threshold : float, optional
        Distance threshold for coloring column dendrogram clusters.
    row_cluster_colors : list of str, optional
        Custom colors for row clusters.
    col_cluster_colors : list of str, optional
        Custom colors for column clusters.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The main figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the heatmap.
    legend_figs : dict
        Dictionary of legend figure objects.
    colorbar_fig : matplotlib.figure.Figure or None
        Separate colorbar figure if separate_colorbar=True.
    patch_colorbar_figs : dict
        Dictionary of separate patch colorbar figures.
    dendrogram_info : dict
        Dictionary containing dendrogram information and cluster assignments.
    
    Examples
    --------
    >>> # Create clustermap with row clustering
    >>> fig, ax, legs, cbar, patch_cbars, dend_info = plot_annotated_clustermap(
    ...     data=data,
    ...     row_cluster=True,
    ...     col_cluster=True,
    ...     row_annotations=row_df,
    ...     row_annotation_col=['category'],
    ...     row_palette=[{'A': 'red', 'B': 'blue'}],
    ...     separate_legends=True
    ... )
    """
    
    # Initialize dendrogram info dictionary
    dendrogram_info = {
        'row_order': None,
        'col_order': None,
        'row_linkage': None,
        'col_linkage': None,
        'row_clusters': None,
        'col_clusters': None
    }
    
    # Prepare dendrogram keyword arguments
    if dendrogram_kws is None:
        dendrogram_kws = {}
    
    if row_dendrogram_kws is None:
        row_dendrogram_kws = {}
    row_dendrogram_kws = {**dendrogram_kws, **row_dendrogram_kws}
    
    if col_dendrogram_kws is None:
        col_dendrogram_kws = {}
    col_dendrogram_kws = {**dendrogram_kws, **col_dendrogram_kws}
    
    # Compute linkage if clustering is requested but linkage not provided
    if row_cluster and row_linkage is None:
        from scipy.cluster.hierarchy import linkage
        row_linkage = linkage(data.values, method='average')
    
    if col_cluster and col_linkage is None:
        from scipy.cluster.hierarchy import linkage
        col_linkage = linkage(data.T.values, method='average')
    
    # Store linkage matrices
    dendrogram_info['row_linkage'] = row_linkage
    dendrogram_info['col_linkage'] = col_linkage
    
    # Reorder data based on linkage if provided
    if row_linkage is not None:
        dendro_row = sch.dendrogram(row_linkage, no_plot=True)
        row_order = dendro_row['leaves']
        data = data.iloc[row_order, :]
        
        if row_annotations is not None:
            row_annotations = row_annotations.iloc[row_order, :]
        
        # Reorder significance data if provided
        if significance_data is not None:
            significance_data = significance_data.iloc[row_order, :]
        
        dendrogram_info['row_order'] = row_order
    
    if col_linkage is not None:
        dendro_col = sch.dendrogram(col_linkage, no_plot=True)
        col_order = dendro_col['leaves']
        data = data.iloc[:, col_order]
        
        if col_annotations is not None:
            col_annotations = col_annotations.iloc[col_order, :]
        
        # Reorder significance data if provided
        if significance_data is not None:
            significance_data = significance_data.iloc[:, col_order]
        
        dendrogram_info['col_order'] = col_order
    
    # Reorder custom tick labels if provided
    if xticklabels is not None:
        if isinstance(xticklabels, pd.Series):
            xticklabels = xticklabels.loc[data.columns].values
        elif col_linkage is not None and dendrogram_info['col_order'] is not None:
            xticklabels = [xticklabels[i] for i in dendrogram_info['col_order']]
    
    if yticklabels is not None:
        if isinstance(yticklabels, pd.Series):
            yticklabels = yticklabels.loc[data.index].values
        elif row_linkage is not None and dendrogram_info['row_order'] is not None:
            yticklabels = [yticklabels[i] for i in dendrogram_info['row_order']]
    
    # Plot dendrograms if axes provided
    if row_dendrogram_ax is not None and row_linkage is not None:
        if row_dendrogram_color == 'cluster':
            if row_color_threshold is None:
                row_color_threshold = 0.7 * np.max(row_linkage[:, 2])
            
            row_dendrogram_kws['color_threshold'] = row_color_threshold
            
            if row_cluster_colors is not None:
                sch.set_link_color_palette(row_cluster_colors)
        else:
            row_dendrogram_kws['color_threshold'] = -np.inf
            row_dendrogram_kws['above_threshold_color'] = row_dendrogram_color
        
        dendro_row = sch.dendrogram(
            row_linkage,
            orientation='right',
            ax=row_dendrogram_ax,
            **row_dendrogram_kws
        )
        
        if row_dendrogram_color == 'cluster':
            from scipy.cluster.hierarchy import fcluster
            row_clusters = fcluster(row_linkage, row_color_threshold, criterion='distance')
            dendrogram_info['row_clusters'] = row_clusters
        
        row_dendrogram_ax.invert_xaxis()
        row_dendrogram_ax.invert_yaxis()
        row_dendrogram_ax.set_yticklabels([])
        row_dendrogram_ax.set_yticks([])
        row_dendrogram_ax.set_xticklabels([])
        row_dendrogram_ax.set_xticks([])
        for spine in row_dendrogram_ax.spines.values():
            spine.set_visible(False)
        
        if row_dendrogram_color == 'cluster' and row_cluster_colors is not None:
            sch.set_link_color_palette(None)
    
    if col_dendrogram_ax is not None and col_linkage is not None:
        if col_dendrogram_color == 'cluster':
            if col_color_threshold is None:
                col_color_threshold = 0.7 * np.max(col_linkage[:, 2])
            
            col_dendrogram_kws['color_threshold'] = col_color_threshold
            
            if col_cluster_colors is not None:
                sch.set_link_color_palette(col_cluster_colors)
        else:
            col_dendrogram_kws['color_threshold'] = -np.inf
            col_dendrogram_kws['above_threshold_color'] = col_dendrogram_color
        
        dendro_col = sch.dendrogram(
            col_linkage,
            ax=col_dendrogram_ax,
            **col_dendrogram_kws
        )
        
        if col_dendrogram_color == 'cluster':
            from scipy.cluster.hierarchy import fcluster
            col_clusters = fcluster(col_linkage, col_color_threshold, criterion='distance')
            dendrogram_info['col_clusters'] = col_clusters
        
        col_dendrogram_ax.set_yticklabels([])
        col_dendrogram_ax.set_yticks([])
        col_dendrogram_ax.set_xticklabels([])
        col_dendrogram_ax.set_xticks([])
        for spine in col_dendrogram_ax.spines.values():
            spine.set_visible(False)
        
        if col_dendrogram_color == 'cluster' and col_cluster_colors is not None:
            sch.set_link_color_palette(None)
    
    # Call the base heatmap function with all parameters
    fig, ax, legend_figs, colorbar_fig, patch_colorbar_figs = plot_annotated_heatmap(
        data=data,
        transpose=transpose,
        fig=fig,
        ax=ax,
        ax_size=ax_size,
        row_annotations=row_annotations,
        col_annotations=col_annotations,
        row_palette=row_palette,
        col_palette=col_palette,
        row_vmin=row_vmin,
        row_vmax=row_vmax,
        row_center=row_center,
        col_vmin=col_vmin,
        col_vmax=col_vmax,
        col_center=col_center,
        row_colorbar_position=row_colorbar_position,
        row_colorbar_size=row_colorbar_size,
        row_colorbar_pad=row_colorbar_pad,
        row_colorbar_label=row_colorbar_label,
        row_cbar_ticks=row_cbar_ticks,
        row_colorbar_orientation=row_colorbar_orientation,
        row_colorbar_coords=row_colorbar_coords,
        row_colorbar_ax=row_colorbar_ax,
        row_separate_colorbar=row_separate_colorbar,
        row_colorbar_figsize=row_colorbar_figsize,
        col_colorbar_position=col_colorbar_position,
        col_colorbar_size=col_colorbar_size,
        col_colorbar_pad=col_colorbar_pad,
        col_colorbar_label=col_colorbar_label,
        col_cbar_ticks=col_cbar_ticks,
        col_colorbar_orientation=col_colorbar_orientation,
        col_colorbar_coords=col_colorbar_coords,
        col_colorbar_ax=col_colorbar_ax,
        col_separate_colorbar=col_separate_colorbar,
        col_colorbar_figsize=col_colorbar_figsize,
        figsize=figsize,
        square=square,
        cmap=cmap,
        font_scale=font_scale,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        row_patch_width=row_patch_width,
        col_patch_height=col_patch_height,
        row_patch_auto_width=row_patch_auto_width,
        col_patch_auto_height=col_patch_auto_height,
        patch_width_ratio=patch_width_ratio,
        patch_height_ratio=patch_height_ratio,
        row_annotation_col=row_annotation_col,
        col_annotation_col=col_annotation_col,
        row_legend_title=row_legend_title,
        col_legend_title=col_legend_title,
        value_legend_title=value_legend_title,
        value_legend_labels=value_legend_labels,
        legend_position=legend_position,
        legend_alignment=legend_alignment,
        legend_bbox_x=legend_bbox_x,
        legend_auto_spacing=legend_auto_spacing,
        legend_spacing=legend_spacing,
        legend_order=legend_order,
        save_path=save_path,
        dpi=dpi,
        show_colorbar=show_colorbar,
        colorbar_position=colorbar_position,
        colorbar_size=colorbar_size,
        colorbar_pad=colorbar_pad,
        colorbar_label=colorbar_label,
        cbar_ticks=cbar_ticks,
        colorbar_orientation=colorbar_orientation,
        colorbar_coords=colorbar_coords,
        colorbar_ax=colorbar_ax,
        separate_colorbar=separate_colorbar,
        colorbar_figsize=colorbar_figsize,
        vmin=vmin,
        vmax=vmax,
        center=center,
        robust=robust,
        heatmap_type=heatmap_type,
        heatmap_linewidths=heatmap_linewidths,
        heatmap_linecolor=heatmap_linecolor,
        patch_linewidths=patch_linewidths,
        patch_linecolor=patch_linecolor,
        legend_patch_linewidths=legend_patch_linewidths,
        legend_patch_linecolor=legend_patch_linecolor,
        row_patch_alpha=row_patch_alpha,
        col_patch_alpha=col_patch_alpha,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        xticklabels_rotation=xticklabels_rotation,
        yticklabels_rotation=yticklabels_rotation,
        xticklabels_position=xticklabels_position,
        yticklabels_position=yticklabels_position,
        auto_tick_padding=auto_tick_padding,
        tick_pad_x=tick_pad_x,
        tick_pad_y=tick_pad_y,
        tick_pad_ratio=tick_pad_ratio,
        base_tick_pad=base_tick_pad,
        font_size_func=font_size_func,
        cbar_kws=cbar_kws,
        row_patch_spacing=row_patch_spacing,
        col_patch_spacing=col_patch_spacing,
        row_separation_col=row_separation_col,
        col_separation_col=col_separation_col,
        row_separation_linewidth=row_separation_linewidth,
        col_separation_linewidth=col_separation_linewidth,
        row_separation_color=row_separation_color,
        col_separation_color=col_separation_color,
        row_separation_linestyle=row_separation_linestyle,
        col_separation_linestyle=col_separation_linestyle,
        row_separation_alpha=row_separation_alpha,
        col_separation_alpha=col_separation_alpha,
        separate_legends=separate_legends,
        legend_orientation=legend_orientation,
        # Pass significance parameters
        significance_data=significance_data,
        significance_func=significance_func,
        significance_marker=significance_marker,
        significance_size_map=significance_size_map,
        significance_color=significance_color,
        significance_alpha=significance_alpha,
        significance_linewidth=significance_linewidth,
        significance_edgecolor=significance_edgecolor,
        significance_text_size=significance_text_size,
        show_significance_legend=show_significance_legend,
        significance_legend_title=significance_legend_title,
        circle_background=circle_background,
    )
    
    # Return with dendrogram info
    return fig, ax, legend_figs, colorbar_fig, patch_colorbar_figs, dendrogram_info

# -----------------------------------------------------------------------
# Annotated Barplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from typing import Optional, Dict, Tuple, Union, Literal, List, Callable


def plot_annotated_barplot(
    data: pd.Series,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    ax_size: Optional[Tuple[float, float]] = None,
    annotations: Optional[pd.DataFrame] = None,
    palette: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    orientation: Literal['horizontal', 'vertical'] = 'horizontal',
    sort_by: Optional[Literal['value', 'index', 'absolute']] = 'value',
    ascending: bool = True,
    top_n: Optional[int] = None,
    title: str = 'Annotated Barplot',
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    bar_colors: Optional[Union[str, List[str], Dict]] = None,
    bar_alpha: float = 0.7,
    bar_edgecolor: str = 'black',
    bar_linewidth: float = 0.5,
    color_by_sign: bool = False,
    positive_color: str = '#e54c38',
    negative_color: str = '#51607f',
    reference_line: Optional[float] = None,
    reference_line_color: str = 'black',
    reference_line_width: float = 1.5,
    reference_line_style: str = '-',
    patch_width: Optional[Union[float, List[float]]] = None,
    patch_auto_width: bool = True,
    patch_width_ratio: float = 0.02,
    patch_spacing: float = 0.005,
    patch_alpha: float = 0.8,
    annotation_col: Optional[Union[str, List[str]]] = None,
    legend_title: Union[str, List[str]] = 'Categories',
    legend_position: Literal['best', 'upper right', 'upper left', 'lower right', 
                             'lower left', 'right', 'center left', 'center right', 
                             'lower center', 'upper center', 'center'] = 'best',
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_orientation: Literal['vertical', 'horizontal'] = 'vertical',
    separate_legends: bool = False,
    show_values: bool = False,
    value_format: str = '.2f',
    value_offset_ratio: float = 0.01,
    significance_values: Optional[Union[pd.Series, Dict, List]] = None,
    significance_thresholds: Optional[List[Tuple[float, str]]] = None,
    significance_symbols: Optional[Dict[str, str]] = None,
    significance_func: Optional[Callable] = None,
    show_ns: bool = False,
    significance_offset_ratio: float = 0.01,
    significance_fontsize: Optional[float] = None,
    significance_color: Optional[Union[str, Literal['bar']]] = 'bar',
    ticklabels: Optional[Union[pd.Series, List, np.ndarray]] = None,
    ticklabels_size: Optional[float] = None,
    symmetric_axis: bool = False,
    axis_padding_ratio: float = 0.3,
    grid: bool = True,
    grid_axis: Literal['both', 'x', 'y'] = 'x',
    grid_alpha: float = 0.3,
    grid_linestyle: str = '--',
    show_spines: Union[bool, Dict[str, bool]] = False,
    font_scale: float = 5,
    font_size_func: Optional[callable] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    patch_spacing_list: Optional[List[float]] = None
) -> Tuple[plt.Figure, plt.Axes, Dict]:
    """
    Create an annotated barplot with colored patches for category annotations.
    Supports both horizontal and vertical orientations.   
    Tick labels and annotation patches are always placed on the OPPOSITE side of the bars.
    
    Parameters
    ----------
    data : pd.Series
        Data to plot as bars. Index contains labels, values are bar heights/lengths.
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, creates a new figure.
        When provided, ax must also be provided.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates a new axes.
        When provided, fig must also be provided.
    ax_size : tuple of float, optional
        Size of the axes (width, height) in inches when using existing fig/ax.
        Used for font size calculations. If None and ax is provided, attempts to
        extract size from axes position, otherwise uses figsize.
    annotations : pd.DataFrame, optional
        DataFrame with bar annotations (index should match data.index).
        Can contain multiple columns for different annotation types.
    palette : dict or list of dict, optional
        Single dictionary or list of dictionaries mapping categories to colors.
        If list, must match length of annotation_col.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, calculated automatically.
        Used when creating new figure or for font size calculations when ax_size is not provided.
    orientation : str, default 'horizontal'
        Orientation of bars: 'horizontal' or 'vertical'. 
    sort_by : str, optional
        How to sort bars: 'value' (by values), 'index' (by index), 
        'absolute' (by absolute values), or None (no sorting).
    ascending : bool, default True
        Sort order if sort_by is specified.
    top_n : int, optional
        Show only top N bars after sorting (None = show all).
    title : str, default 'Annotated Barplot'
        Title for the plot.
    xlabel : str, optional
        Label for x-axis. If None, auto-generated based on orientation.
    ylabel : str, optional
        Label for y-axis. If None, auto-generated based on orientation.
    bar_colors : str, list, or dict, optional
        Colors for bars. Can be:
        - Single color (str): all bars same color
        - List of colors: one per bar
        - Dict mapping indices to colors
        Ignored if color_by_sign=True.  
    bar_alpha : float, default 0.7
        Transparency of bars (0.0-1.0).
    bar_edgecolor : str, default 'black'
        Color of bar edges.
    bar_linewidth : float, default 0.5
        Width of bar edges.  
    color_by_sign : bool, default False
        If True, color bars by sign (positive vs negative).
    positive_color : str, default '#e54c38'
        Color for positive values (when color_by_sign=True).
    negative_color : str, default '#51607f'
        Color for negative values (when color_by_sign=True).
    reference_line : float, optional
        Value at which to draw a reference line (e.g., 0 for fold changes).
    reference_line_color : str, default 'black'
        Color of reference line.
    reference_line_width : float, default 1.5
        Width of reference line.
    reference_line_style : str, default '-'
        Style of reference line ('-', '--', '-.', ':').
    patch_width : float or list of float, optional
        Manual width(s) of annotation patches. If None and auto is True, calculated automatically.
        If list, must match length of annotation_col.
    patch_auto_width : bool, default True
        Automatically calculate patch width based on axis dimensions.
    patch_width_ratio : float, default 0.02
        Ratio of axis range for patches (used when auto width is True).
    patch_spacing : float, default 0.005
        Spacing between patches and axis/labels as ratio of axis range.
    patch_alpha : float, default 0.8
        Transparency of annotation patches (0.0-1.0).
    annotation_col : str or list of str, optional
        Column name(s) in annotations DataFrame to use for coloring.
        If list, creates multiple annotation patches per bar.
    legend_title : str or list of str, default 'Categories'
        Title(s) for annotation legend(s).
        If list, must match length of annotation_col.  
    legend_position : str, default 'best'
        Position of legend.   
    legend_bbox_to_anchor : tuple, optional
        Bbox to anchor for legend positioning.
    legend_orientation : str, default 'vertical'
        Orientation of legend items: 'vertical' (stacked) or 'horizontal' (side-by-side).
        When 'horizontal', all items in a legend are displayed in one row.
    separate_legends : bool, default False
        If True, create separate figure objects for each legend instead of placing them on the main figure.
        Useful for complex layouts or when legends need independent positioning.
    show_values : bool, default False
        Whether to show value labels at the end of bars.
    value_format : str, default '.2f'
        Format string for value labels.
    value_offset_ratio : float, default 0.01
        Offset for value labels as ratio of axis range.
    significance_values : pd.Series, dict, or list, optional
        Significance values (e.g., p-values) for each bar.
        Can be:
        - pd.Series with same index as data
        - Dict mapping indices to significance values
        - List with same length as data
    significance_thresholds : list of tuples, optional
        Thresholds for significance levels as (threshold, symbol) tuples.
        Default: [(1e-4, '****'), (1e-3, '***'), (1e-2, '**'), (0.05, '*')]
        Thresholds are evaluated in order (most to least significant).
    significance_symbols : dict, optional
        Custom mapping of threshold labels to display symbols.
        Overrides symbols from significance_thresholds.
    significance_func : callable, optional
        Custom function to convert significance values to symbols.
        Should accept a float and return a string.
        If provided, overrides significance_thresholds. 
    show_ns : bool, default False
        Whether to show 'ns' (not significant) for non-significant values.
    significance_offset_ratio : float, default 0.01
        Offset for significance symbols from bar ends as ratio of axis range.
    significance_fontsize : float, optional
        Font size for significance symbols. If None, uses 'text' size from font_scale.
    significance_color : str or 'bar', default 'bar'
        Color for significance symbols. If 'bar', uses the same color as the bar.
    ticklabels : pd.Series, list, or np.ndarray, optional
        Custom labels for bars. If None, uses data.index.
        Must have the same length as data (after filtering/sorting).
    ticklabels_size : float, optional
        Font size for tick labels. If None, uses default from font_scale.
    symmetric_axis : bool, default False
        Make value axis symmetric around reference_line (useful for fold changes).
    axis_padding_ratio : float, default 0.3
        Padding at the end of value axis as ratio of data range.
    grid : bool, default True
        Whether to show grid.   
    grid_axis : str, default 'x'
        Which axis to show grid: 'x', 'y', or 'both'.
    grid_alpha : float, default 0.3
        Transparency of grid lines.
    grid_linestyle : str, default '--'
        Style of grid lines.
    show_spines : bool or dict, default False
        Whether to show axis spines. Can be bool or dict with keys 'top', 'right', 'bottom', 'left'.
    font_scale : float, default 5
        Scale factor for font sizes.
    font_size_func : callable, optional
        Custom function to calculate font sizes. Should accept (width, height, unit, scale)
        and return a dict with keys: 'ticks_label', 'label', 'legend', 'legend_title', 'title', 'text'.
    save_path : str, optional
        Path to save the figure. If separate_legends is True, individual legend files
        will be saved with '_legend_{key}' appended to the filename.
    dpi : int, default 600
        DPI for saved figure.
    patch_spacing_list : list of float, optional
        Custom spacing for each annotation patch level. If None, uses patch_spacing for all.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.  
    ax : matplotlib.axes.Axes
        The axes object.
    legend_figs : dict
        Dictionary of legend figure objects if separate_legends=True, otherwise empty dict.
        Keys are legend identifiers (e.g., 'annot_0', 'annot_1').
    
    Examples
    --------
    >>> # Basic usage - creates new figure
    >>> fig, ax, legends = plot_annotated_barplot(
    ...     data=fold_changes,
    ...     annotations=df_annotations,
    ...     annotation_col='pathway',
    ...     palette={'Metabolic': 'red', 'Signaling': 'blue'},
    ...     color_by_sign=True,
    ...     reference_line=0
    ... )
    
    >>> # Using existing figure and axes
    >>> fig, axes = plt.subplots(2, 1, figsize=(15, 20))
    >>> plot_annotated_barplot(
    ...     data=data1,
    ...     fig=fig, ax=axes[0],
    ...     ax_size=(14, 9),
    ...     title='Dataset 1'
    ... )
    >>> plot_annotated_barplot(
    ...     data=data2,
    ...     fig=fig, ax=axes[1],
    ...     ax_size=(14, 9),
    ...     title='Dataset 2'
    ... )
    
    >>> # With separate legends
    >>> fig, ax, legends = plot_annotated_barplot(
    ...     data=fold_changes,
    ...     annotations=df_annotations,
    ...     annotation_col=['pathway', 'tissue'],
    ...     palette=[palette1, palette2],
    ...     separate_legends=True,
    ...     legend_orientation='horizontal',
    ...     save_path='barplot.png'
    ... )
    """
    
    # Validate fig and ax parameters
    if (fig is None) != (ax is None):
        raise ValueError("Both fig and ax must be provided together, or both must be None")
    
    # Determine if we're using existing axes
    use_existing_ax = (fig is not None and ax is not None)
    
    # Process data
    plot_data = data.copy()
    
    # Sort data if requested
    if sort_by == 'value':
        plot_data = plot_data.sort_values(ascending=ascending)
    elif sort_by == 'absolute':
        plot_data = plot_data.iloc[np.abs(plot_data.values).argsort()]
        if not ascending:
            plot_data = plot_data.iloc[::-1]
    elif sort_by == 'index':
        plot_data = plot_data.sort_index(ascending=ascending)
    
    # Filter top N if requested
    if top_n is not None:
        plot_data = plot_data.iloc[-top_n:] if ascending else plot_data.iloc[:top_n]
    
    # Get bar labels and values
    bar_labels = plot_data.index.values
    bar_values = plot_data.values
    n_bars = len(bar_labels)
    
    # Process significance values
    if significance_values is not None:
        if isinstance(significance_values, pd.Series):
            sig_vals = significance_values.loc[bar_labels].values
        elif isinstance(significance_values, dict):
            sig_vals = [significance_values.get(label, np.nan) for label in bar_labels]
        else:
            sig_vals = list(significance_values)
            assert len(sig_vals) == n_bars, "significance_values must match data length"
    else:
        sig_vals = None
    
    # Setup default significance thresholds
    if significance_thresholds is None:
        significance_thresholds = [
            (1e-4, '****'),
            (1e-3, '***'),
            (1e-2, '**'),
            (0.05, '*')
        ]
    
    # Create significance conversion function
    if significance_func is None:
        def default_sig_func(val):
            if pd.isna(val):
                return ''
            for threshold, symbol in significance_thresholds:
                if val <= threshold:
                    return symbol
            return 'ns' if show_ns else ''
        
        sig_func = default_sig_func
    else:
        sig_func = significance_func
    
    # Process custom tick labels
    if ticklabels is not None:
        if isinstance(ticklabels, pd.Series):
            ticklabels = ticklabels.loc[bar_labels].values
        else:
            ticklabels = list(ticklabels)
        assert len(ticklabels) == n_bars, "ticklabels must match data length"
    else:
        ticklabels = bar_labels
    
    # Convert annotation parameters to lists
    if annotation_col is not None and not isinstance(annotation_col, list):
        annotation_col = [annotation_col]
    if palette is not None and not isinstance(palette, list):
        palette = [palette]
    if isinstance(legend_title, str):
        legend_title = [legend_title]
    if patch_width is not None and not isinstance(patch_width, list):
        patch_width = [patch_width]
    
    # Validate annotations
    if annotation_col is not None:
        n_annots = len(annotation_col)
        if palette is not None:
            assert len(palette) == n_annots, "palette must match annotation_col length"
        if len(legend_title) == 1:
            legend_title = legend_title * n_annots
        assert len(legend_title) == n_annots, "legend_title must match annotation_col length"
        if patch_width is not None:
            if len(patch_width) == 1:
                patch_width = patch_width * n_annots
            assert len(patch_width) == n_annots, "patch_width must match annotation_col length"
        
        # Ensure annotations match data
        if annotations is not None:
            annotations = annotations.loc[bar_labels]
    
    # Setup patch spacing
    if patch_spacing_list is None:
        patch_spacing_list = [patch_spacing] * (len(annotation_col) if annotation_col else 1)
    
    # Calculate figure size
    if figsize is None:
        if orientation == 'horizontal':
            figsize = (14, max(6, n_bars * 0.6))
        else:
            figsize = (max(8, n_bars * 0.6), 10)
    
    # Determine size for font calculations
    if use_existing_ax:
        if ax_size is not None:
            calc_width, calc_height = ax_size
        else:
            bbox = ax.get_position()
            fig_width, fig_height = fig.get_size_inches()
            calc_width = bbox.width * fig_width
            calc_height = bbox.height * fig_height
    else:
        calc_width, calc_height = figsize
    
    # Calculate font sizes
    if font_size_func is not None:
        font_size = font_size_func(calc_width, calc_height, 'in', font_scale, 'number')
    else:
        # Default font sizes
        font_size = {
            'ticks_label': 8,
            'label': 10,
            'legend': 9,
            'legend_title': 10,
            'title': 12,
            'text': 9
        }
    
    # Override tick label size if specified
    if ticklabels_size is not None:  
        font_size['ticks_label'] = ticklabels_size
    
    # Set significance font size
    if significance_fontsize is None:
        significance_fontsize = font_size['text']
    
    # Create figure if not provided
    if not use_existing_ax:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Determine bar colors
    if color_by_sign:
        colors = [negative_color if x < 0 else positive_color for x in bar_values]
    elif bar_colors is not None:
        if isinstance(bar_colors, str):
            colors = [bar_colors] * n_bars
        elif isinstance(bar_colors, dict):
            colors = [bar_colors.get(label, 'gray') for label in bar_labels]
        else:
            colors = bar_colors
    else:
        colors = ['steelblue'] * n_bars
    
    # Create barplot
    positions = np.arange(n_bars)
    if orientation == 'horizontal':
        bars = ax.barh(positions, bar_values, height=1,
                       color=colors, alpha=bar_alpha,
                       edgecolor=bar_edgecolor, linewidth=bar_linewidth)
        value_axis = 'x'
        category_axis = 'y'
    else:
        bars = ax.bar(positions, bar_values, width=1,
                      color=colors, alpha=bar_alpha,
                      edgecolor=bar_edgecolor, linewidth=bar_linewidth)
        value_axis = 'y'
        category_axis = 'x'
    
    # Set axis limits with padding
    if symmetric_axis and reference_line is not None:  
        max_abs_val = np.abs(bar_values).max()
        axis_limit = max_abs_val * (1 + axis_padding_ratio)
        if orientation == 'horizontal':
            ax.set_xlim(-axis_limit, axis_limit)
            axis_range = 2 * axis_limit
        else:
            ax.set_ylim(-axis_limit, axis_limit)
            axis_range = 2 * axis_limit
    else:
        val_min, val_max = bar_values.min(), bar_values.max()
        val_range = val_max - val_min
        padding = val_range * axis_padding_ratio
        
        # Extend limits to include reference line if specified
        if reference_line is not None:
            val_min = min(val_min, reference_line)
            val_max = max(val_max, reference_line)
        
        if orientation == 'horizontal':
            ax.set_xlim(val_min - padding, val_max + padding)
            axis_range = val_max - val_min + 2 * padding
        else:
            ax.set_ylim(val_min - padding, val_max + padding)
            axis_range = val_max - val_min + 2 * padding
    
    # Calculate patch dimensions
    legend_dict = {}
    
    if annotation_col is not None and annotations is not None and palette is not None:
        n_annots = len(annotation_col)
        if patch_width is None:
            patch_width = [None] * n_annots
        
        for idx in range(n_annots):
            if patch_auto_width and patch_width[idx] is None:   
                calculated_width = axis_range * patch_width_ratio
                patch_width[idx] = max(calculated_width, axis_range * 0.01)
            elif patch_width[idx] is None:
                patch_width[idx] = axis_range * 0.02
        
        # Calculate total patch width including spacing
        total_patch_width = sum(patch_width) + sum([patch_spacing_list[i] * axis_range for i in range(n_annots - 1)])
        
        # Add annotation patches - ALWAYS on opposite side of bars
        for annot_idx in range(n_annots):
            # Calculate starting position for this annotation (from outermost to innermost)
            start_offset = sum(patch_width[:annot_idx]) + sum([patch_spacing_list[i] * axis_range for i in range(annot_idx)])
            
            col_name = annotation_col[annot_idx]
            present_categories = set()
            
            for i, bar_label in enumerate(bar_labels):
                category = annotations.loc[bar_label, col_name]
                present_categories.add(category)
                color = palette[annot_idx].get(category, 'gray')
                
                bar_val = bar_values[i]
                
                if orientation == 'horizontal':
                    # Patch on OPPOSITE side of bar
                    if bar_val >= 0:
                        # Bar goes right, patch on left
                        patch_x = -(total_patch_width - start_offset + patch_spacing_list[0] * axis_range)
                    else:
                        # Bar goes left, patch on right
                        patch_x = patch_spacing_list[0] * axis_range + start_offset
                    
                    rect = Rectangle(
                        (patch_x, i - 0.4),
                        patch_width[annot_idx],
                        0.8,
                        facecolor=color,
                        edgecolor=bar_edgecolor,
                        linewidth=bar_linewidth,
                        alpha=patch_alpha,
                        clip_on=False,
                        zorder=10
                    )
                else:
                    # Vertical orientation - patch on OPPOSITE side of bar
                    if bar_val >= 0:
                        # Bar goes up, patch on bottom
                        patch_y = -(total_patch_width - start_offset + patch_spacing_list[0] * axis_range)
                    else:
                        # Bar goes down, patch on top
                        patch_y = patch_spacing_list[0] * axis_range + start_offset
                    
                    rect = Rectangle(
                        (i - 0.4, patch_y),
                        0.8,
                        patch_width[annot_idx],
                        facecolor=color,
                        edgecolor=bar_edgecolor,
                        linewidth=bar_linewidth,
                        alpha=patch_alpha,
                        clip_on=False,
                        zorder=10
                    )
                
                ax.add_patch(rect)
            
            # Create legend for this annotation
            legend_elements = [
                Patch(facecolor=color, edgecolor=bar_edgecolor, 
                     linewidth=bar_linewidth, label=category, alpha=patch_alpha)
                for category, color in palette[annot_idx].items()
                if category in present_categories
            ]
            
            legend_dict[f'annot_{annot_idx}'] = {
                'handles': legend_elements,
                'title': legend_title[annot_idx],
                'n_items': len(legend_elements)
            }
        
        # Calculate label offset (beyond patches)
        label_offset = total_patch_width + 2 * patch_spacing_list[0] * axis_range
    else:
        label_offset = patch_spacing * axis_range
    
    # Add tick labels - ALWAYS on opposite side of bars
    if orientation == 'horizontal':
        ax.set_yticks(positions)
        
        for i, (label, val) in enumerate(zip(ticklabels, bar_values)):
            # Label on OPPOSITE side of bar
            if val >= 0:
                # Bar goes right, label on left
                x_pos = -label_offset
                ha = 'right'
            else:
                # Bar goes left, label on right
                x_pos = label_offset
                ha = 'left'
            
            ax.text(x_pos, i, f" {label} ",
                   fontsize=font_size['ticks_label'], ha=ha, va='center', zorder=11)
        
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)  # Hide tick marks
    else:
        ax.set_xticks(positions)
        
        for i, (label, val) in enumerate(zip(ticklabels, bar_values)):
            # Label on OPPOSITE side of bar
            if val >= 0:
                # Bar goes up, label on bottom
                y_pos = -label_offset
                va = 'top'
            else:
                # Bar goes down, label on top
                y_pos = label_offset
                va = 'bottom'
            
            ax.text(i, y_pos, f" {label} ",
                   fontsize=font_size['ticks_label'], ha='center', va=va, 
                   rotation=45, zorder=11)
        
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)  # Hide tick marks
    
    # Add significance symbols at bar ends
    if sig_vals is not None:
        sig_offset = axis_range * significance_offset_ratio
        
        for i, (val, sig_val) in enumerate(zip(bar_values, sig_vals)):
            symbol = sig_func(sig_val)
            
            if symbol and symbol != 'ns' or (symbol == 'ns' and show_ns):
                # Determine symbol color
                if significance_color == 'bar': 
                    sym_color = colors[i]
                else: 
                    sym_color = significance_color
                
                if orientation == 'horizontal':
                    # Position symbol at bar end
                    if val >= 0:
                        x_pos = val + sig_offset
                        ha = 'left'
                    else:
                        x_pos = val - sig_offset
                        ha = 'right'
                    
                    ax.text(x_pos, i, symbol,
                           fontsize=significance_fontsize, ha=ha, va='center',
                           color=sym_color, zorder=12)
                else: 
                    # Vertical orientation
                    if val >= 0:
                        y_pos = val + sig_offset
                        va = 'bottom'
                    else:
                        y_pos = val - sig_offset
                        va = 'top'
                    
                    ax.text(i, y_pos, symbol,
                           fontsize=significance_fontsize, ha='center', va=va,
                           color=sym_color, zorder=12)
    
    # Add value labels at bar ends if requested
    if show_values:  
        value_offset = axis_range * value_offset_ratio
        
        # Adjust offset if significance symbols are shown
        if sig_vals is not None:
            value_offset += axis_range * significance_offset_ratio * 2
        
        for i, (val, bar) in enumerate(zip(bar_values, bars)):
            if orientation == 'horizontal':
                if val >= 0:
                    x_pos = val + value_offset
                    ha = 'left'
                else:
                    x_pos = val - value_offset
                    ha = 'right'
                ax.text(x_pos, i, f"{val:{value_format}}",
                       fontsize=font_size['text'], ha=ha, va='center',
                       color=colors[i])
            else:
                if val >= 0:
                    y_pos = val + value_offset
                    va = 'bottom'
                else:
                    y_pos = val - value_offset
                    va = 'top'
                ax.text(i, y_pos, f"{val:{value_format}}",
                       fontsize=font_size['text'], ha='center', va=va,
                       color=colors[i])
    
    # Add reference line
    if reference_line is not None:
        if orientation == 'horizontal':
            ax.axvline(x=reference_line, color=reference_line_color,
                      linestyle=reference_line_style, linewidth=reference_line_width,
                      zorder=5)
        else:
            ax.axhline(y=reference_line, color=reference_line_color,
                      linestyle=reference_line_style, linewidth=reference_line_width,
                      zorder=5)
    
    # Add grid
    if grid:  
        ax.grid(axis=grid_axis, alpha=grid_alpha, linestyle=grid_linestyle, zorder=0)
    
    # Set axis labels
    if xlabel is None:
        xlabel = 'Value' if orientation == 'horizontal' else 'Category'
    if ylabel is None:  
        ylabel = 'Category' if orientation == 'horizontal' else 'Value'
    
    ax.set_xlabel(xlabel, fontsize=font_size['label'])
    ax.set_ylabel(ylabel, fontsize=font_size['label'])
    
    # Set title
    ax.set_title(title, fontsize=font_size['title'], pad=20)
    
    # Configure spines
    if isinstance(show_spines, bool):
        for spine in ax.spines.values():
            spine.set_visible(show_spines)
    else:
        for spine_name, visible in show_spines.items():
            if spine_name in ax.spines:
                ax.spines[spine_name].set_visible(visible)
    
    # Handle legends
    legends_to_plot = list(legend_dict.keys())
    legend_figs = {}
    
    def get_ncol(n_items, orientation):
        return n_items if orientation == 'horizontal' else 1
    
    # CASE A: Separate Legends
    if separate_legends:
        for key in legends_to_plot:
            legend_info = legend_dict[key]
            n_items = legend_info['n_items']
            
            if legend_orientation == 'horizontal':
                figsize_legend = (n_items * 1.5, 1) 
            else:
                figsize_legend = (2, n_items * 0.5 + 0.5)
                
            l_fig = plt.figure(figsize=figsize_legend)
            l_ax = l_fig.add_subplot(111)
            l_ax.axis('off')
            
            l_ax.legend(
                handles=legend_info['handles'],
                title=legend_info['title'],
                loc='center',
                ncol=get_ncol(n_items, legend_orientation),
                frameon=False,
                fontsize=font_size.get('legend', 10),
                title_fontsize=font_size.get('legend_title', 12)
            )
            legend_figs[key] = l_fig
            
            if save_path is not None:
                base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, 'png')
                l_save_path = f"{base}_legend_{key}.{ext}"
                l_fig.savefig(l_save_path, dpi=dpi, bbox_inches='tight')
    
    # CASE B: Legends on Main Figure
    else:
        all_handles = []
        
        for key in legends_to_plot:
            legend_info = legend_dict[key]
            all_handles.extend(legend_info['handles'])
        
        if all_handles:
            if legend_bbox_to_anchor:  
                ax.legend(
                    handles=all_handles, 
                    loc=legend_position,
                    bbox_to_anchor=legend_bbox_to_anchor,
                    ncol=get_ncol(len(all_handles), legend_orientation),
                    fontsize=font_size['legend'],
                    title_fontsize=font_size.get('legend_title', font_size['legend']),
                    frameon=True
                )
            else:
                ax.legend(
                    handles=all_handles, 
                    loc=legend_position,
                    ncol=get_ncol(len(all_handles), legend_orientation),
                    fontsize=font_size['legend'],
                    title_fontsize=font_size.get('legend_title', font_size['legend']),
                    frameon=True
                )
    
    # Only call tight_layout if we created the figure
    if not use_existing_ax:
        plt.tight_layout()
    
    # Save figure (only if path provided and we created the figure)
    if save_path is not None and not use_existing_ax:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=False)
    
    return fig, ax, legend_figs