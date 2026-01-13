import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D

def plot_dim_reduction(
    coords_df,
    dr_method="auto",  # Method name for labeling (or "auto" to infer)
    coords_cols=None,  # List of coordinate columns to plot, e.g. ['PC1', 'PC2']
    explained=None,
    color_col=None,
    style_col=None,
    size=100,
    figsize=(8, 6),
    title="Dimensionality Reduction",
    font_sizes=None,
    legend_title=None,
    color_palette=None,  # NEW: Optional dict mapping category values to colors
    show_loadings=False,
    loadings_df=None,
    loadings_scale=1,
    arrow_color='k',
    arrow_width=0.01,
    arrow_headwidth=0.02,
    show_centroids=False,
    centroid_categories=None,
    centroid_size=200,
    centroid_marker='X',
    show_ellipses=False,
    ellipse_std=1,
    ellipse_alpha=0.2,
    filepath=None,
    show=True,
):
    """
    General-purpose dimensionality reduction plot for PCA/PCoA/tSNE/UMAP/etc.

    - coords_df: dataframe containing embedding results
    - dr_method: 'PCA', 'PCoA', 'tSNE', 'UMAP', or 'auto' (tries to infer from col names, for axis labeling)
    - coords_cols: list of columns to use for coordinates; must be length 2, e.g. ['PC1','PC2'] or ['UMAP1','UMAP2']
    - explained: list of explained variances for axes (optional, mainly for PCA/PCoA)
    - color_palette: dict mapping category values to colors (e.g.  {'GroupA': '#FF0000', 'GroupB': '#0000FF'})
                     If None, uses default seaborn 'Set2' palette
    """
    if coords_cols is None: 
        # Try to infer coordinates:  prefer first two columns matching known DR axes
        candidates = ['PC1','PC2','PCoA1','PCoA2','Dim1','Dim2','UMAP1','UMAP2','tSNE1','tSNE2']
        found = [c for c in candidates if c in coords_df. columns]
        if len(found) >= 2:
            coords_cols = found[:2]
        else: 
            coords_cols = coords_df.columns[: 2].tolist()
    x, y = coords_cols
    if dr_method == "auto": 
        # Pick first coordinate name prefix if available
        prefixes = ['PC', 'PCoA', 'UMAP', 'tSNE', 'Dim']
        dr_method = next((p for p in prefixes if x.startswith(p)), "DimRed")
    # Compose axis labels using explained variance if present
    def axis_label(col, idx):
        label = col
        if explained is not None and idx < len(explained):
            label += f" ({explained[idx]*100:.2f}%)"
        return label
    x_label = axis_label(x, 0)
    y_label = axis_label(y, 1)
    legend_title = legend_title or f"{color_col or ''}{', ' if color_col and style_col else ''}{style_col or ''}"

    fig, ax = plt.subplots(figsize=figsize)

    # Style and color mappings
    marker_symbols = ['o', 'X', '^', 's', 'D', 'v', 'P', '*', '<', '>', '8', 'p', 'H', 'h', '+', 'x']
    style_map = {}
    if style_col is not None: 
        unique_styles = coords_df[style_col].dropna().unique()
        style_map = {val: marker_symbols[i % len(marker_symbols)] for i, val in enumerate(unique_styles)}
        coords_df['_marker_'] = coords_df[style_col].map(style_map)
    else:
        coords_df['_marker_'] = 'o'
    
    # Color palette - use custom palette if provided, otherwise generate default
    unique_colors = coords_df[color_col].dropna().unique() if color_col else ["__single__"]
    if color_palette is not None: 
        # Use provided color palette dictionary
        color_map = color_palette.copy()
        # Fill in any missing categories with default colors
        missing_cats = [cat for cat in unique_colors if cat not in color_map]
        if missing_cats:
            default_palette = sns.color_palette('Set2', n_colors=len(missing_cats))
            for cat, col in zip(missing_cats, default_palette):
                color_map[cat] = col
    else:
        # Generate default palette
        palette = sns.color_palette('Set2', n_colors=len(unique_colors))
        color_map = dict(zip(unique_colors, palette)) if color_col else {"__single__": "gray"}
    
    coords_df['_color_'] = coords_df[color_col].map(color_map) if color_col else "gray"

    # Scatter by color/style
    for i, (grp_label, grp_data) in enumerate(coords_df.groupby(color_col) if color_col else [("__single__", coords_df)]):
        for style_val, style_grp in grp_data.groupby(style_col) if style_col else [("__single__", grp_data)]:
            marker = style_map.get(style_val, 'o')
            color = color_map.get(grp_label, "gray")
            ax.scatter(
                style_grp[x], style_grp[y],
                marker=marker,
                c=[color],
                edgecolor='k',
                alpha=0.5,
                s=size,
                label=None
            )

    ax.set_title(title, fontsize=font_sizes.get('title',14) if font_sizes else 14, pad=20)
    ax.set_xlabel(x_label, fontsize=font_sizes. get('axes_label',12) if font_sizes else 12)
    ax.set_ylabel(y_label, fontsize=font_sizes.get('axes_label',12) if font_sizes else 12)
    ax.tick_params(axis='both', which='major', labelsize=font_sizes.get('ticks_label',10) if font_sizes else 10)

    ax.axhline(0, color='grey', lw=1, ls='--')
    ax.axvline(0, color='grey', lw=1, ls='--')

    # ---- Centroid/Ellipse ----
    ellipse_legend_elements = []
    if show_centroids or show_ellipses: 
        if centroid_categories is None:
            centroid_categories = color_col
        if isinstance(centroid_categories, str):
            centroid_categories = [centroid_categories]
        coords_df['_group_'] = coords_df[centroid_categories].astype(str).agg('_'.join, axis=1)
        group_labels = coords_df['_group_'].unique()
        
        # Use custom palette for ellipses if color_col matches centroid_categories
        if color_palette is not None and len(centroid_categories) == 1 and centroid_categories[0] == color_col: 
            group_color_map = color_map.copy()
            # Preserve order from unique_colors for consistent legend ordering
            ordered_groups = [g for g in unique_colors if g in group_labels]
        else:
            palette_ellipse = sns.color_palette('Set2', n_colors=len(group_labels))
            group_color_map = dict(zip(group_labels, palette_ellipse))
            ordered_groups = group_labels
        
        for group_name in ordered_groups:
            group_data = coords_df[coords_df['_group_'] == group_name]
            if len(group_data) == 0:
                continue
                
            x_vals = group_data[x].values
            y_vals = group_data[y].values
            centroid_x = np.mean(x_vals)
            centroid_y = np.mean(y_vals)
            
            # Get color from appropriate map
            if len(centroid_categories) == 1 and centroid_categories[0] == color_col:
                group_color = color_map.get(group_data[color_col].iloc[0], 'gray')
            else:
                group_color = group_color_map[group_name]
            
            group_label = group_data[centroid_categories[0]].iloc[0] if len(centroid_categories) == 1 else group_name
            if show_ellipses and len(x_vals) > 2:
                cov = np.cov(x_vals, y_vals)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width = 2 * ellipse_std * np.sqrt(eigenvalues[0])
                height = 2 * ellipse_std * np.sqrt(eigenvalues[1])
                ellipse = Ellipse(
                    xy=(centroid_x, centroid_y),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=group_color,
                    edgecolor=group_color,
                    alpha=ellipse_alpha,
                    linestyle='--',
                    linewidth=2
                )
                ax.add_patch(ellipse)
                # Append in order (no sorting needed later)
                ellipse_legend_elements.append((group_label, Patch(facecolor=group_color, edgecolor=group_color, alpha=ellipse_alpha, label=str(group_label))))
            if show_centroids:
                ax.scatter(
                    centroid_x, 
                    centroid_y,
                    s=centroid_size,
                    marker=centroid_marker,
                    c=[group_color],
                    edgecolors='black',
                    linewidths=2,
                    zorder=10,
                    alpha=0.9
                )
        coords_df.drop('_group_', axis=1, inplace=True)

    # ---- Loadings ----
    if show_loadings and loadings_df is not None:
        for idx, row in loadings_df.iterrows():
            ax.arrow(
                0, 0, 
                row[x]*loadings_scale, 
                row[y]*loadings_scale, 
                color=arrow_color, 
                width=arrow_width, 
                head_width=arrow_headwidth, 
                alpha=0.8
            )
            ax.text(
                row[x]*(loadings_scale+0.15), 
                row[y]*(loadings_scale+0.15), 
                str(idx), 
                color=arrow_color, 
                fontsize=font_sizes. get('loadings',12) if font_sizes else 11, 
                ha='center', 
                va='center'
            )

    # ---- Legend ----
    legend_handles = []
    legend_labels = []
    # Color legend
    if color_col is not None:
        legend_handles.append(Line2D([0], [0], color='none'))
        legend_labels. append('Color Style')
        for val in unique_colors:
            legend_handles.append(Patch(facecolor=color_map[val], edgecolor='k', label=str(val), alpha=0.5))
            legend_labels.append(str(val))
    # Style legend
    if style_col is not None:  
        legend_handles.append(Line2D([0], [0], color='none'))
        legend_labels.append('Shape Style')
        for style in unique_styles:
            marker = style_map.get(style, 'o')
            legend_handles.append(Line2D([0], [0], marker=marker, color='w', 
                                         markerfacecolor='grey', 
                                         markeredgecolor='k', markeredgewidth=1, 
                                         markersize=10, linestyle='', label=str(style)))
            legend_labels.append(str(style))
    # Ellipse legend - NO SORTING, maintain order from loop
    if ellipse_legend_elements:
        legend_handles.append(Line2D([0], [0], color='none'))
        legend_labels.append('Ellipsoid')
        for lab, handle in ellipse_legend_elements:  # Removed sorted()
            legend_handles.append(handle)
            legend_labels.append(lab)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        title=legend_title,
        title_fontsize=font_sizes.get('legend_title',12) if font_sizes else 12,
        fontsize=font_sizes.get('legend',10) if font_sizes else 10,
        loc='center left', 
        bbox_to_anchor=(1.1, 0.5)
    )
    plt.tight_layout()

    
    if filepath and show:
        plt.savefig(filepath, dpi=600, bbox_inches='tight', transparent=False)
        plt.show()
    elif filepath and not show:
        plt.savefig(filepath, dpi=600, bbox_inches='tight', transparent=False)
        plt.close()
    elif not filepath and show:
        plt.show()
    elif not filepath and not show:
        return fig, ax



# ---------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Sequence, Dict, Union

def plot_dim_reduction_plotly(
    coords_df: pd.DataFrame,
    x: str = "PCoA1",
    y: str = "PCoA2",
    z: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    z_label: Optional[str] = None,
    explained: Optional[Sequence[float]] = None,
    color_col: Optional[str] = None,
    style_col: Optional[str] = None,
    size_col: Optional[str] = None,
    size: Union[int, float] = 8,
    title: str = "Dimensionality reduction",
    font_sizes: Optional[Dict[str, int]] = None,
    legend_title: Optional[str] = None,
    hover_cols: Optional[Sequence[str]] = None,
    show_loadings: bool = False,
    loadings_df: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    loadings_scale: Optional[float] = None,
    show_only_top_loadings: Optional[int] = None,
    arrow_color: str = "black",
    arrow_width: float = 3.0,
    save_html: bool = False,
    html_path: str = "dim_reduction_plot.html",
    save_png: bool = False,
    png_path: str = "dim_reduction_plot.png",
    ):
    """
    Create a 2D or 3D interactive dimensionality-reduction scatter with Plotly.

    Returns the plotly Figure (also shows it in notebooks).
    """
    font_sizes = font_sizes or {}
    hover_cols = list(hover_cols) if hover_cols is not None else []

    # sanity checks
    if x not in coords_df.columns or y not in coords_df.columns:
        raise ValueError(f"x and y must be columns in coords_df. Available: {list(coords_df.columns)}")
    is_3d = z is not None
    if is_3d and z not in coords_df.columns:
        raise ValueError("z specified but not found in coords_df")

    # axis label helper to append explained variance when possible
    def _append_expl(axis_name: str, label_override: Optional[str]) -> str:
        label = label_override or axis_name
        if explained is not None:
            import re
            m = re.search(r"(\d+)(?!.*\d)", axis_name)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(explained):
                    label = f"{label} ({100.0*float(explained[idx]):.2f}%)"
        return label

    x_label = _append_expl(x, x_label)
    y_label = _append_expl(y, y_label)
    z_label = _append_expl(z, z_label) if is_3d else None

    # legend title
    if legend_title is None:
        pieces = [p for p in (color_col, style_col) if p]
        legend_title = ", ".join(pieces) if pieces else ""

    # build hover_data (avoid including coords twice)
    exclude = {x, y}
    if is_3d: exclude.add(z)
    hover_data = []
    for c in hover_cols:
        if c in coords_df.columns and c not in hover_data:
            hover_data.append(c)
    for c in coords_df.columns:
        if c not in exclude and c not in hover_data:
            hover_data.append(c)

    # prepare px kwargs
    px_kwargs = dict(
        data_frame=coords_df,
        color=color_col if (color_col in coords_df.columns) else None,
        symbol=style_col if (style_col in coords_df.columns) else None,
        hover_name=coords_df.index.astype(str),
        hover_data=hover_data,
        title=title,
    )

    # size handling: if size_col given use it; else we'll set a constant marker size after creation
    use_size_col = (size_col is not None and size_col in coords_df.columns)
    if use_size_col:
        px_kwargs['size'] = size_col

    if is_3d:
        px_kwargs.update(dict(x=x, y=y, z=z))
        fig = px.scatter_3d(**px_kwargs)
    else:
        px_kwargs.update(dict(x=x, y=y))
        fig = px.scatter(**px_kwargs)

    # set a constant marker size if not using size_col
    if not use_size_col:
        # update only the scatter traces produced by px (not future loadings traces)
        fig.update_traces(marker=dict(size=size), selector=dict(mode="markers"))

    # layout fonts/sizes
    base_font = int(font_sizes.get('axes_label', 12))
    tick_font = int(font_sizes.get('ticks_label', max(base_font - 2, 9)))
    title_font = int(font_sizes.get('title', max(base_font + 2, 14)))
    legend_font = int(font_sizes.get('legend', max(base_font - 1, 10)))
    legend_title_font = int(font_sizes.get('legend_title', legend_font))

    if is_3d:
        fig.update_layout(
            scene=dict(
                xaxis=dict(title=dict(text=x_label, font=dict(size=base_font)), tickfont=dict(size=tick_font)),
                yaxis=dict(title=dict(text=y_label, font=dict(size=base_font)), tickfont=dict(size=tick_font)),
                zaxis=dict(title=dict(text=z_label, font=dict(size=base_font)), tickfont=dict(size=tick_font)),
            ),
            title=dict(text=title, font=dict(size=title_font)),
            legend=dict(title=dict(text=legend_title), font=dict(size=legend_font)),
            margin=dict(l=0, r=0, b=0, t=60),
        )
    else:
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=dict(text=title, font=dict(size=title_font)),
            legend=dict(title=dict(text=legend_title), font=dict(size=legend_font)),
            margin=dict(l=40, r=220, b=40, t=60),
        )
        # add faint 0 lines
        fig.add_shape(type="line", x0=0, x1=0, y0=coords_df[y].min(), y1=coords_df[y].max(),
                      line=dict(color="lightgrey", width=1, dash="dash"))
        fig.add_shape(type="line", x0=coords_df[x].min(), x1=coords_df[x].max(), y0=0, y1=0,
                      line=dict(color="lightgrey", width=1, dash="dash"))

    # handle loadings (if requested)
    if show_loadings and loadings_df is not None:
        # normalize loadings into DataFrame
        if isinstance(loadings_df, np.ndarray):
            load_df = pd.DataFrame(loadings_df.copy(),
                                   index=[f"feat_{i}" for i in range(loadings_df.shape[0])],
                                   columns=[f"Comp{i+1}" for i in range(loadings_df.shape[1])])
        elif isinstance(loadings_df, pd.DataFrame):
            load_df = loadings_df.copy()
        else:
            load_df = None

        if load_df is None:
            print("loadings_df not recognized; skipping loadings.")
        else:
            needed = 3 if is_3d else 2
            if load_df.shape[1] < needed:
                print(f"loadings_df must have at least {needed} columns. Skipping loadings.")
            else:
                # parse component indices from axis names if possible, otherwise default 0/1/2
                def _comp_index_from_axis(axis_name: str, default: int) -> int:
                    import re
                    if not axis_name:
                        return default
                    m = re.search(r"(\d+)(?!.*\d)", axis_name)
                    if m:
                        idx = int(m.group(1)) - 1
                        if 0 <= idx < load_df.shape[1]:
                            return idx
                    return default

                xi = _comp_index_from_axis(x, 0)
                yi = _comp_index_from_axis(y, 1)
                zi = _comp_index_from_axis(z, 2) if is_3d else None

                comps = load_df.iloc[:, [xi, yi]] if not is_3d else load_df.iloc[:, [xi, yi, zi]]
                mags = np.linalg.norm(comps.values, axis=1)
                order = np.argsort(mags)[::-1]
                if show_only_top_loadings is not None:
                    order = order[:show_only_top_loadings]

                # automatic scale if not provided
                if loadings_scale is None:
                    coords_for_span = coords_df[[x, y]] if not is_3d else coords_df[[x, y, z]]
                    coord_span = np.ptp(coords_for_span.values, axis=0)
                    coord_span[coord_span == 0] = 1.0
                    max_span = np.max(coord_span)
                    max_loading = mags.max() if mags.max() != 0 else 1.0
                    loadings_scale_calc = 0.65 * max_span / max_loading
                else:
                    loadings_scale_calc = loadings_scale

                # add traces for each loading (lines and text)
                for idx in order:
                    feat = load_df.index[idx]
                    vec = comps.values[idx]
                    if not is_3d:
                        x0, y0 = 0.0, 0.0
                        x1, y1 = vec[0] * loadings_scale_calc, vec[1] * loadings_scale_calc
                        fig.add_trace(go.Scatter(
                            x=[x0, x1],
                            y=[y0, y1],
                            mode='lines+markers',
                            marker=dict(size=2, color=arrow_color),
                            line=dict(color=arrow_color, width=arrow_width),
                            hoverinfo='text',
                            text=[str(feat)],
                            showlegend=False,
                        ))
                        fig.add_trace(go.Scatter(
                            x=[x1],
                            y=[y1],
                            mode='text',
                            text=[str(feat)],
                            textposition='top center',
                            showlegend=False,
                        ))
                    else:
                        x0, y0, z0 = 0.0, 0.0, 0.0
                        x1, y1, z1 = vec[0] * loadings_scale_calc, vec[1] * loadings_scale_calc, vec[2] * loadings_scale_calc
                        fig.add_trace(go.Scatter3d(
                            x=[x0, x1],
                            y=[y0, y1],
                            z=[z0, z1],
                            mode='lines+markers',
                            marker=dict(size=2, color=arrow_color),
                            line=dict(color=arrow_color, width=arrow_width),
                            hoverinfo='text',
                            text=[str(feat)],
                            showlegend=False,
                        ))
                        fig.add_trace(go.Scatter3d(
                            x=[x1],
                            y=[y1],
                            z=[z1],
                            mode='text',
                            text=[str(feat)],
                            textposition='top center',
                            showlegend=False,
                        ))

    # final layout tweaks and show/save
    fig.update_layout(font=dict(size=int(font_sizes.get('axes_label', 12))))
    fig.show()

    if save_html:
        try:
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Saved interactive HTML to {html_path}")
        except Exception as e:
            print("Failed to save HTML:", e)

    if save_png:
        try:
            fig.write_image(png_path)
            print(f"Saved PNG to {png_path}")
        except Exception as e:
            print("Failed to save PNG (kaleido may be required):", e)

    return fig


# ---------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib. patches import Rectangle
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from typing import Optional, Dict, Tuple, Union, Literal


def plot_annotated_heatmap(
    data: pd. DataFrame,
    row_annotations:  Optional[pd.DataFrame] = None,
    col_annotations: Optional[pd.DataFrame] = None,
    row_palette: Optional[Dict[str, str]] = None,
    col_palette: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 14),
    cmap: Union[str, LinearSegmentedColormap] = 'Blues',
    font_scale: float = 13,
    title: str = 'Annotated Heatmap',
    xlabel: str = 'Columns',
    ylabel: str = 'Rows',
    row_patch_width: Optional[float] = None,
    col_patch_height: Optional[float] = None,
    row_patch_auto_width: bool = True,
    col_patch_auto_height: bool = True,
    patch_width_ratio: float = 0.05,
    patch_height_ratio: float = 0.05,
    row_annotation_col: Optional[str] = None,
    col_annotation_col: Optional[str] = None,
    row_legend_title: str = 'Row Categories',
    col_legend_title: str = 'Column Categories',
    value_legend_title:  str = 'Values',
    value_legend_labels: Optional[Dict] = None,
    legend_position: Literal['right', 'left', 'top', 'bottom'] = 'right',
    legend_alignment: Literal['top', 'center', 'bottom'] = 'top',
    legend_bbox_x: float = 1.02,
    legend_auto_spacing: bool = True,
    legend_spacing: float = 0.08,
    legend_order: Optional[list] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    show_colorbar:  bool = False,
    colorbar_position:  Literal['right', 'left', 'top', 'bottom'] = 'right',
    colorbar_size: str = '3%',
    colorbar_pad: float = 0.05,
    colorbar_label:  Optional[str] = None,
    colorbar_orientation: Optional[Literal['vertical', 'horizontal']] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    robust: bool = False,
    heatmap_type: Literal['qualitative', 'quantitative'] = 'qualitative',
    linewidths: float = 0.5,
    linecolor: str = 'grey',
    xticklabels_rotation: float = 45,
    yticklabels_rotation: float = 0,
    auto_tick_padding: bool = True,
    tick_pad_x: Optional[float] = None,
    tick_pad_y: Optional[float] = None,
    tick_pad_ratio: float = 1.5,
    base_tick_pad: float = 5.0,
    font_size_func: Optional[callable] = None,
    cbar_kws: Optional[Dict] = None
    ) -> Tuple[plt.Figure, plt. Axes]:
    """
    Create an annotated heatmap with colored patches for row and column categories. 
    Supports both qualitative (binary/categorical) and quantitative (gradient) heatmaps.
    
    Parameters
    ----------
    data :  pd.DataFrame
        The main data to plot as a heatmap. 
    row_annotations : pd.DataFrame, optional
        DataFrame with row annotations (index should match data. index).
    col_annotations : pd.DataFrame, optional
        DataFrame with column annotations (index should match data.columns).
    row_palette : dict, optional
        Dictionary mapping row categories to colors.
    col_palette : dict, optional
        Dictionary mapping column categories to colors.
    figsize :  tuple, default (10, 14)
        Figure size (width, height) in inches.
    cmap :  str or colormap, default 'Blues'
        Colormap for the heatmap.  Popular options: 
        - Quantitative: 'viridis', 'plasma', 'RdYlBu_r', 'coolwarm', 'seismic'
        - Qualitative:  'Blues', 'Reds', 'Greens'
    font_scale : float, default 13
        Scale factor for font sizes.
    title : str, default 'Annotated Heatmap'
        Title for the plot.
    xlabel : str, default 'Columns'
        Label for x-axis.
    ylabel : str, default 'Rows'
        Label for y-axis. 
    row_patch_width : float, optional
        Manual width of row annotation patches.  If None and auto is True, calculated automatically.
    col_patch_height : float, optional
        Manual height of column annotation patches. If None and auto is True, calculated automatically. 
    row_patch_auto_width : bool, default True
        Automatically calculate row patch width based on heatmap dimensions.
    col_patch_auto_height : bool, default True
        Automatically calculate column patch height based on heatmap dimensions.
    patch_width_ratio : float, default 0.05
        Ratio of heatmap width for row patches (used when auto width is True).
    patch_height_ratio : float, default 0.05
        Ratio of heatmap height for column patches (used when auto height is True).
    row_annotation_col : str, optional
        Column name in row_annotations to use for coloring.
    col_annotation_col : str, optional
        Column name in col_annotations to use for coloring.
    row_legend_title : str, default 'Row Categories'
        Title for row annotation legend.
    col_legend_title : str, default 'Column Categories'
        Title for column annotation legend.
    value_legend_title : str, default 'Values'
        Title for value legend (used for qualitative heatmaps).
    value_legend_labels : dict, optional
        Custom labels for value legend (e.g., {'high': 'Present', 'low': 'Absent'}).
        Only used when heatmap_type='qualitative'.
    legend_position :  str, default 'right'
        Position of legends: 'right', 'left', 'top', or 'bottom'.
    legend_alignment : str, default 'top'
        Alignment of legend group: 'top', 'center', or 'bottom'.
    legend_bbox_x : float, default 1.02
        X-coordinate for legend bounding box (for right/left positions).
    legend_auto_spacing : bool, default True
        Automatically calculate spacing between legends based on their sizes.
    legend_spacing : float, default 0.08
        Vertical spacing between legends (used if auto_spacing is False).
    legend_order : list, optional
        Order of legends.  Can contain 'row', 'col', 'value'. 
        Default is ['row', 'col', 'value'].
    save_path :  str, optional
        Path to save the figure. 
    dpi : int, default 600
        DPI for saved figure.
    show_colorbar : bool, default False
        Whether to show colorbar for heatmap values.  Automatically True for quantitative heatmaps.
    colorbar_position : str, default 'right'
        Position of colorbar: 'right', 'left', 'top', or 'bottom'.
    colorbar_size : str, default '3%'
        Size of the colorbar (percentage of the axes size).
    colorbar_pad :  float, default 0.05
        Padding between heatmap and colorbar.
    colorbar_label : str, optional
        Label for the colorbar. 
    colorbar_orientation : str, optional
        Orientation of colorbar.  If None, automatically determined from position.
    vmin : float, optional
        Minimum value for colorbar normalization.
    vmax : float, optional
        Maximum value for colorbar normalization.
    center :  float, optional
        Value at which to center the colormap (for diverging colormaps).
    robust : bool, default False
        If True, use robust quantiles for colorbar limits.
    heatmap_type : str, default 'qualitative'
        Type of heatmap: 'qualitative' (binary/categorical) or 'quantitative' (gradient).
    linewidths : float, default 0.5
        Width of lines between cells.
    linecolor : str, default 'grey'
        Color of lines between cells.
    xticklabels_rotation :  float, default 45
        Rotation angle for x-axis tick labels.
    yticklabels_rotation : float, default 0
        Rotation angle for y-axis tick labels.
    auto_tick_padding : bool, default True
        Automatically calculate tick padding based on patch dimensions.
    tick_pad_x : float, optional
        Manual padding for x-axis ticks.  If None and auto is True, calculated automatically. 
    tick_pad_y : float, optional
        Manual padding for y-axis ticks. If None and auto is True, calculated automatically. 
    tick_pad_ratio : float, default 1.5
        Multiplier for converting patch dimensions to tick padding (in points).
        Higher values create more space between patches and tick labels.
    base_tick_pad : float, default 5.0
        Base padding (in points) added to calculated tick padding.
    font_size_func : callable, optional
        Custom function to calculate font sizes. Should accept (width, height, unit, scale)
        and return a dict with keys:  'ticks_label', 'label', 'legend', 'legend_title', 'title'.
    cbar_kws : dict, optional
        Additional keyword arguments for colorbar customization. 
    
    Returns
    -------
    fig : matplotlib.figure. Figure
        The figure object. 
    ax : matplotlib.axes. Axes
        The axes object. 
    
    Examples
    --------
    >>> # Automatic tick padding based on patches
    >>> fig, ax = plot_annotated_heatmap(
    ...     data=expression_data,
    ...     row_annotations=df_row_info,
    ...     col_annotations=df_col_info,
    ...     heatmap_type='quantitative',
    ...     auto_tick_padding=True,
    ...     tick_pad_ratio=2.0  # More space
    ... )
    
    >>> # Manual tick padding
    >>> fig, ax = plot_annotated_heatmap(
    ...     data=expression_data,
    ...     auto_tick_padding=False,
    ...     tick_pad_x=15,
    ...     tick_pad_y=25
    ... )
    """
    
    # Calculate font sizes
    if font_size_func is not None:
        font_size = font_size_func(figsize[0], figsize[1], 'in', scale=font_scale)
    else:
        # Default font sizes
        font_size = {
            'ticks_label':  8,
            'label': 10,
            'legend': 9,
            'legend_title':  10,
            'title': 12,
            'colorbar_label': 10,
            'colorbar_ticks': 8
        }
    
    # Ensure indices match if annotations are provided
    if row_annotations is not None:
        row_annotations = row_annotations.loc[data.index]
    if col_annotations is not None: 
        col_annotations = col_annotations.loc[data.columns]
    
    # Automatically enable colorbar for quantitative heatmaps
    if heatmap_type == 'quantitative': 
        show_colorbar = True
    
    # Determine colorbar orientation
    if colorbar_orientation is None:
        if colorbar_position in ['right', 'left']: 
            colorbar_orientation = 'vertical'
        else:
            colorbar_orientation = 'horizontal'
    
    # Create figure and axis
    fig, ax = plt. subplots(figsize=figsize)
    
    # Prepare colorbar keyword arguments
    if cbar_kws is None: 
        cbar_kws = {}
    
    # Set colorbar position and orientation
    if show_colorbar:
        cbar_kws. update({
            'orientation': colorbar_orientation,
            'pad': colorbar_pad,
            'label':  colorbar_label if colorbar_label else ''
        })
        
        # Add fraction (size) parameter
        if 'fraction' not in cbar_kws:
            # Convert percentage string to float
            if isinstance(colorbar_size, str) and '%' in colorbar_size: 
                size_value = float(colorbar_size. rstrip('%')) / 100
            else: 
                size_value = 0.03
            cbar_kws['fraction'] = size_value
    
    # Plot the heatmap using Seaborn
    im = sns.heatmap(
        data,
        cmap=cmap,
        ax=ax,
        cbar=show_colorbar,
        cbar_kws=cbar_kws if show_colorbar else None,
        linewidths=linewidths,
        linecolor=linecolor,
        xticklabels=True,
        yticklabels=True,
        vmin=vmin,
        vmax=vmax,
        center=center,
        robust=robust
    )
    
    # Customize colorbar if shown
    if show_colorbar and hasattr(im, 'collections') and len(im.collections) > 0:
        # Get the colorbar
        cbar = ax.collections[0].colorbar
        
        # Customize colorbar label
        if colorbar_label:
            cbar.set_label(colorbar_label,
                          fontsize=font_size. get('cbar_label', font_size['label']),
                          rotation=90 if colorbar_orientation == 'vertical' else 0,
                          labelpad=10)
        
        # Customize colorbar tick labels
        cbar.ax.tick_params(labelsize=font_size.get('cbar_ticks', font_size['ticks_label']))
        
        # Add frame around colorbar
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor('black')
    
    # Customize tick labels
    ax.set_xticklabels(
        data.columns,
        rotation=xticklabels_rotation,
        ha='right',
        fontsize=font_size['ticks_label']
    )
    ax.set_yticklabels(
        data.index,
        rotation=yticklabels_rotation,
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
    
    # Calculate automatic patch dimensions
    if row_patch_auto_width and row_patch_width is None: 
        # Calculate width based on heatmap width
        calculated_row_patch_width = heatmap_width * patch_width_ratio
        # Ensure minimum visibility (at least 0.3 units)
        row_patch_width = max(calculated_row_patch_width, 0.3)
    elif row_patch_width is None: 
        row_patch_width = 0.5  # default
    
    if col_patch_auto_height and col_patch_height is None:
        # Calculate height based on heatmap height
        calculated_col_patch_height = heatmap_height * patch_height_ratio
        # Ensure minimum visibility (at least 1 unit)
        col_patch_height = max(calculated_col_patch_height, 1.0)
    elif col_patch_height is None:
        col_patch_height = 2.0  # default
    
    # Calculate automatic tick padding based on patch dimensions
    if auto_tick_padding:
        # Get figure DPI for conversion
        fig_dpi = fig. dpi
        
        # Convert patch dimensions from data coordinates to points
        # For y-axis (row patches): width in data coords -> points
        if tick_pad_y is None and row_annotations is not None and row_annotation_col is not None:
            # Calculate width in inches based on figure size and axis position
            bbox = ax.get_position()
            ax_width_inches = bbox.width * figsize[0]
            
            # Data units per inch
            data_per_inch_x = heatmap_width / ax_width_inches
            
            # Patch width in inches
            patch_width_inches = row_patch_width / data_per_inch_x
            
            # Convert to points (1 inch = 72 points)
            patch_width_points = patch_width_inches * 35
            
            # Calculate padding with ratio and base
            tick_pad_y = base_tick_pad + (patch_width_points * tick_pad_ratio)
        elif tick_pad_y is None: 
            tick_pad_y = base_tick_pad
        
        # For x-axis (column patches): height in data coords -> points
        if tick_pad_x is None and col_annotations is not None and col_annotation_col is not None:
            # Calculate height in inches based on figure size and axis position
            bbox = ax.get_position()
            ax_height_inches = bbox.height * figsize[1]
            
            # Data units per inch
            data_per_inch_y = heatmap_height / ax_height_inches
            
            # Patch height in inches
            patch_height_inches = col_patch_height / data_per_inch_y
            
            # Convert to points (1 inch = 72 points)
            patch_height_points = - patch_height_inches * 35
            
            # Calculate padding with ratio and base
            tick_pad_x = base_tick_pad + (patch_height_points * tick_pad_ratio)
        elif tick_pad_x is None:
            tick_pad_x = base_tick_pad
    else:
        # Use manual padding or defaults
        if tick_pad_x is None:
            tick_pad_x = 20
        if tick_pad_y is None:
            tick_pad_y = 20
    
    # Adjust tick parameters with calculated padding
    ax.tick_params(axis='x', which='both', length=0, pad=tick_pad_x)
    ax.tick_params(axis='y', which='both', length=0, pad=tick_pad_y)
    
    # Add spines around the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('black')
    
    # Dictionary to store legend information with keys
    legend_dict = {}
    
    # Add patches for rows (left side)
    if row_annotations is not None and row_annotation_col is not None and row_palette is not None:
        for i, row_idx in enumerate(data.index):
            category = row_annotations.loc[row_idx, row_annotation_col]
            color = row_palette. get(category, 'grey')
            rect = Rectangle(
                (xlim[0] - row_patch_width, i),
                row_patch_width,
                1,
                linewidth=linewidths,
                edgecolor=linecolor,
                facecolor=color,
                clip_on=False
            )
            ax.add_patch(rect)
        
        # Create legend for row categories
        legend_elements_row = [
            Rectangle((0, 0), 1, 1, fc=color, ec=linecolor, linewidth=linewidths, label=category)
            for category, color in row_palette.items()
        ]
        legend_dict['row'] = {
            'handles': legend_elements_row,
            'title': row_legend_title,
            'n_items': len(legend_elements_row)
        }
    
    # Add patches for columns (bottom)
    if col_annotations is not None and col_annotation_col is not None and col_palette is not None:
        for j, col_idx in enumerate(data.columns):
            category = col_annotations.loc[col_idx, col_annotation_col]
            color = col_palette.get(category, 'grey')
            rect = Rectangle(
                (j, ylim[0]),
                1,
                col_patch_height,
                linewidth=linewidths,
                edgecolor=linecolor,
                facecolor=color,
                clip_on=False
            )
            ax.add_patch(rect)
        
        # Create legend for column categories
        legend_elements_col = [
            Rectangle((0, 0), 1, 1, fc=color, ec=linecolor, linewidth=linewidths, label=category)
            for category, color in col_palette.items()
        ]
        legend_dict['col'] = {
            'handles':  legend_elements_col,
            'title': col_legend_title,
            'n_items':  len(legend_elements_col)
        }
    
    # Add value legend if specified (only for qualitative heatmaps)
    if value_legend_labels is not None and heatmap_type == 'qualitative':
        # Get colormap colors
        if isinstance(cmap, str):
            colormap = plt.get_cmap(cmap)
        else:
            colormap = cmap
        
        # Create legend elements based on value_legend_labels
        if 'custom' in value_legend_labels: 
            # Custom legend elements provided
            legend_elements_value = value_legend_labels['custom']
        else:
            # Create default binary legend (e.g., Present/Absent)
            legend_elements_value = [
                Rectangle((0, 0), 1, 1, fc=colormap(1.0), ec=linecolor,
                         linewidth=linewidths, label=value_legend_labels. get('high', 'High')),
                Rectangle((0, 0), 1, 1, fc=colormap(0.0), ec=linecolor,
                         linewidth=linewidths, label=value_legend_labels. get('low', 'Low'))
            ]
        
        legend_dict['value'] = {
            'handles': legend_elements_value,
            'title': value_legend_title,
            'n_items': len(legend_elements_value)
        }
    
    # Determine legend order
    if legend_order is None:
        legend_order = ['row', 'col', 'value']
    
    # Filter legend_order to only include legends that exist
    legends_to_plot = [key for key in legend_order if key in legend_dict]
    
    # Calculate automatic spacing if enabled
    if legend_auto_spacing and legends_to_plot:
        # Estimate legend heights based on number of items
        # Each item is approximately 0.03 in figure coordinates, title adds 0.04
        legend_heights = []
        for key in legends_to_plot: 
            n_items = legend_dict[key]['n_items']
            estimated_height = 0.04 + (n_items * 0.03)  # title + items
            legend_heights.append(estimated_height)
        
        total_legend_height = sum(legend_heights) + (len(legends_to_plot) - 1) * legend_spacing
        
        # Calculate starting position based on alignment
        if legend_alignment == 'top':
            start_y = 0.95
        elif legend_alignment == 'center':
            start_y = 0.5 + (total_legend_height / 2)
        elif legend_alignment == 'bottom':
            start_y = 0.05 + total_legend_height
        else: 
            start_y = 0.95  # default to top
        
        # Calculate positions for each legend
        legend_positions = []
        current_y = start_y
        for height in legend_heights:
            legend_positions.append(current_y - height / 2)
            current_y -= (height + legend_spacing)
    else:
        # Use manual spacing
        if legend_alignment == 'top':
            start_y = 0.95
        elif legend_alignment == 'center':
            start_y = 0.5
        elif legend_alignment == 'bottom':
            start_y = 0.05 + (len(legends_to_plot) - 1) * legend_spacing
        else:
            start_y = 0.95
        
        legend_positions = [start_y - i * legend_spacing for i in range(len(legends_to_plot))]
    
    # Adjust legend position if colorbar is shown on the same side
    if show_colorbar and colorbar_position == legend_position:
        if legend_position == 'right':
            legend_bbox_x = legend_bbox_x + 0.15  # Move legends further right
        elif legend_position == 'left':
            legend_bbox_x = legend_bbox_x - 0.15  # Move legends further left
    
    # Set bbox_to_anchor based on position
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
    
    # Add all legends to the figure
    for idx, key in enumerate(legends_to_plot):
        legend_info = legend_dict[key]
        y_position = legend_positions[idx]
        
        fig.legend(
            handles=legend_info['handles'],
            title=legend_info['title'],
            loc=legend_loc,
            bbox_to_anchor=(bbox_x, y_position),
            frameon=True,
            fontsize=font_size['legend'],
            title_fontsize=font_size['legend_title'],
        )
    
    # Set title
    ax.set_title(title, fontsize=font_size['title'], pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=False)
    
    return fig, ax
