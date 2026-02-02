import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib. patches import Ellipse, Patch
from matplotlib.lines import Line2D

def plot_dim_reduction(
    coords_df,
    dr_method="auto",  # Method name for labeling (or "auto" to infer)
    coords_cols=None,  # List of coordinate columns to plot, e. g. ['PC1', 'PC2']
    explained=None,
    color_col=None,
    style_col=None,
    size=100,
    figsize=(8, 6),
    title="Dimensionality Reduction",
    font_sizes=None,
    legend_title=None,
    separate_legends=False,  # NEW: Whether to create separate legend figures
    legend_orientation='vertical',  # NEW: 'vertical' or 'horizontal' for legends
    color_palette=None,  # NEW:  Optional dict mapping category values to colors
    show_loadings=False,
    loadings_df=None,
    loadings_scale=1,
    arrow_color='k',
    arrow_width=0.05,
    arrow_headwidth=0.2,
    show_centroids=False,
    centroid_categories=None,
    centroid_size=200,
    centroid_marker='X',
    show_ellipses=False,
    ellipse_std=1,
    ellipse_alpha=0.2,
    show_labels=False,  # NEW: Whether to show point labels
    label_col=None,  # NEW: Column name containing labels to display
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
    - show_labels: if True, display text labels for each point
    - label_col: column name in coords_df to use for labels (if None, uses index)
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
        palette = sns. color_palette('Set2', n_colors=len(unique_colors))
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
                
            x_vals = group_data[x]. values
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
        loading_texts = []
        loadings_fontsize = font_sizes.get('loadings', 12) if font_sizes else 11
        
        for idx, row in loadings_df.iterrows():
            # Draw Arrow
            ax.arrow(
                0, 0, 
                row[x]*loadings_scale, 
                row[y]*loadings_scale, 
                color=arrow_color, 
                width=arrow_width, 
                head_width=arrow_headwidth, 
                alpha=0.8
            )
            
            # Draw Text at initial position
            # We assign it to a variable 'txt' and append to list
            txt = ax.text(
                (row[x]*loadings_scale)+0.8 if row[x]>=0 else (row[x]*loadings_scale)-0.8, 
                (row[y]*loadings_scale)+0.8 if row[y]>=0 else (row[y]*loadings_scale)-0.8, 
                str(idx), 
                color=arrow_color, 
                fontsize=loadings_fontsize, 
                ha='center', 
                va='center'
            )
            loading_texts.append(txt)
            
    # ---- Point Labels ----
    if show_labels: 
        try:
            from adjustText import adjust_text
            texts = []
            label_fontsize = font_sizes.get('text', 10) if font_sizes else 10
            
            # Use label_col if provided, otherwise use index
            if label_col is not None and label_col in coords_df.columns:
                labels = coords_df[label_col]
            else:
                labels = coords_df.index
            
            for idx, row in coords_df.iterrows():
                label = labels[idx] if label_col else idx
                text = ax.text(
                    row[x], 
                    row[y], 
                    str(label),
                    fontsize=label_fontsize,
                    ha='center',
                    va='center'
                )
                texts.append(text)
            
            # Adjust text positions to avoid overlap
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7),
                ax=ax
            )
        except ImportError: 
            # Fallback if adjustText is not installed - simple offset labels
            label_fontsize = font_sizes. get('text', 10) if font_sizes else 10
            
            if label_col is not None and label_col in coords_df.columns:
                labels = coords_df[label_col]
            else:
                labels = coords_df.index
            
            for idx, row in coords_df.iterrows():
                label = labels[idx] if label_col else idx
                ax.text(
                    row[x], 
                    row[y], 
                    str(label),
                    fontsize=label_fontsize,
                    ha='left',
                    va='bottom',
                    alpha=0.8
                )

    # # ---- Legend ----
    # legend_handles = []
    # legend_labels = []
    # # Color legend
    # if color_col is not None: 
    #     legend_handles.append(Line2D([0], [0], color='none'))
    #     legend_labels.append('Color Style')
    #     for val in unique_colors:
    #         legend_handles.append(Patch(facecolor=color_map[val], edgecolor='k', label=str(val), alpha=0.5))
    #         legend_labels.append(str(val))
    # # Style legend
    # if style_col is not None:  
    #     legend_handles.append(Line2D([0], [0], color='none'))
    #     legend_labels.append('Shape Style')
    #     for style in unique_styles:
    #         marker = style_map.get(style, 'o')
    #         legend_handles.append(Line2D([0], [0], marker=marker, color='w', 
    #                                      markerfacecolor='grey', 
    #                                      markeredgecolor='k', markeredgewidth=1, 
    #                                      markersize=10, linestyle='', label=str(style)))
    #         legend_labels.append(str(style))
    # # Ellipse legend - NO SORTING, maintain order from loop
    # if ellipse_legend_elements:
    #     legend_handles.append(Line2D([0], [0], color='none'))
    #     legend_labels.append('Ellipsoid')
    #     for lab, handle in ellipse_legend_elements:   # Removed sorted()
    #         legend_handles.append(handle)
    #         legend_labels. append(lab)

    # fig.legend(
    #     handles=legend_handles,
    #     labels=legend_labels,
    #     title=legend_title,
    #     title_fontsize=font_sizes. get('legend_title',12) if font_sizes else 12,
    #     fontsize=font_sizes.get('legend',10) if font_sizes else 10,
    #     loc='center left', 
    #     bbox_to_anchor=(1.1, 0.5)
    # )
    # plt.tight_layout()

    
    # if filepath and show:
    #     plt.savefig(filepath, dpi=600, bbox_inches='tight', transparent=False)
    #     plt.show()
    # elif filepath and not show:
    #     plt.savefig(filepath, dpi=600, bbox_inches='tight', transparent=False)
    #     plt.close()
    # elif not filepath and show:
    #     plt.show()
    # elif not filepath and not show:
    #     return fig, ax
    # ---- Legend Logic ----
    legend_figs = {} # Dictionary to store separate legend figures
    
    # 1. Collect handles into groups first
    # This keeps the logic clean regardless of whether we want separate or combined legends
    legend_groups = {} 
    
    if color_col is not None:
        handles = []
        for val in unique_colors:
            handles.append(Patch(facecolor=color_map[val], edgecolor='k', label=str(val), alpha=0.5))
        legend_groups['Color Style'] = handles

    if style_col is not None:
        handles = []
        for style in unique_styles:
            marker = style_map.get(style, 'o')
            handles.append(Line2D([0], [0], marker=marker, color='w', 
                                         markerfacecolor='grey', 
                                         markeredgecolor='k', markeredgewidth=1, 
                                         markersize=10, linestyle='', label=str(style)))
        legend_groups['Shape Style'] = handles
        
    if ellipse_legend_elements:
        handles = []
        # No sorting, maintain order from loop as per original code
        for lab, handle in ellipse_legend_elements:
            handle.set_label(lab) # Ensure label is attached to handle
            handles.append(handle)
        legend_groups['Ellipsoid'] = handles

    # 2. Generate Legends based on configuration
    if separate_legends:
        # --- OPTION A: Separate Figures ---
        for title, handles in legend_groups.items():
            n_items = len(handles)
            
            # Calculate dynamic figsize based on orientation
            if legend_orientation == 'horizontal':
                ncol = n_items
                # Width grows with items, fixed height
                figsize_leg = (n_items * 1.5 + 1, 1.2) 
            else:
                ncol = 1
                # Fixed width, height grows with items
                figsize_leg = (2.5, n_items * 0.4 + 0.8)
            
            l_fig = plt.figure(figsize=figsize_leg)
            l_ax = l_fig.add_subplot(111)
            l_ax.axis('off')
            
            l_ax.legend(
                handles=handles,
                title=title,
                loc='center',
                ncol=ncol,
                frameon=False,
                fontsize=font_sizes.get('legend', 10) if font_sizes else 10,
                title_fontsize=font_sizes.get('legend_title', 12) if font_sizes else 12
            )
            
            # Store in dict
            clean_key = title.lower().replace(' ', '_')
            legend_figs[clean_key] = l_fig
            
            # Save immediately if filepath provided
            if filepath:
                base, ext = filepath.rsplit('.', 1)
                l_path = f"{base}_legend_{clean_key}.{ext}"
                l_fig.savefig(l_path, dpi=600, bbox_inches='tight', transparent=False)

    else:
        # --- OPTION B: Combined Legend (Main Figure) ---
        legend_handles = []
        legend_labels = []
        
        for title, handles in legend_groups.items():
            # Add Section Header (Invisible line hack)
            legend_handles.append(Line2D([0], [0], color='none'))
            legend_labels.append(title)
            
            # Add actual items
            for h in handles:
                legend_handles.append(h)
                legend_labels.append(h.get_label())

        # Only add legend if items exist
        if legend_handles:
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
    
    # 3. Handle Output
    if filepath and show:
        plt.savefig(filepath, dpi=600, bbox_inches='tight', transparent=False)
        plt.show()
    elif filepath and not show:
        plt.savefig(filepath, dpi=600, bbox_inches='tight', transparent=False)
        plt.close()
    elif not filepath and show:
        plt.show()
    
    # Always return figures if not just showing
    if not show:
        return fig, ax, legend_figs

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
# Annotated Heatmap with Row/Column Categories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from typing import Optional, Dict, Tuple, Union, Literal, List, Sequence


def plot_annotated_heatmap(
    data: pd.DataFrame,
    transpose: bool = False,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    ax_size: Optional[Tuple[float, float]] = None,
    row_annotations: Optional[pd.DataFrame] = None,
    col_annotations: Optional[pd.DataFrame] = None,
    row_palette: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
    col_palette: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
    figsize: Tuple[float, float] = (10, 14),
    square: bool = False,
    cmap: Union[str, LinearSegmentedColormap] = 'Blues',
    font_scale: float = 13,
    title: str = 'Annotated Heatmap',
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
    linewidths: float = 0.5,
    linecolor: str = 'grey',
    row_patch_alpha: float = 1.0,
    col_patch_alpha: float = 1.0,
    xticklabels: Optional[Union[pd.Series, List, np.ndarray]] = None,
    yticklabels: Optional[Union[pd.Series, List, np.ndarray]] = None,
    xticklabels_rotation: float = 45,
    yticklabels_rotation: float = 0,
    auto_tick_padding: bool = True,
    tick_pad_x: Optional[float] = None,
    tick_pad_y: Optional[float] = None,
    tick_pad_ratio: float = 1.5,
    base_tick_pad: float = 5.0,
    font_size_func: Optional[callable] = None,
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
) -> Tuple[plt.Figure, plt.Axes, Dict, Optional[plt.Figure]]:
    """
    Create an annotated heatmap with colored patches for row and column categories. 
    Supports both qualitative (binary/categorical) and quantitative (gradient) heatmaps.
    Now supports multiple annotation columns from a single DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        The main data to plot as a heatmap.
    transpose : bool, default False
        If True, transpose the data matrix and swap all row/column related parameters.
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
    row_annotations : pd.DataFrame, optional
        Single DataFrame with row annotations (index should match data.index).
        Can contain multiple columns for different annotation types.
    col_annotations : pd.DataFrame, optional
        Single DataFrame with column annotations (index should match data.columns).
        Can contain multiple columns for different annotation types.
    row_palette : dict or list of dict, optional
        Single dictionary or list of dictionaries mapping row categories to colors.
        If list, must match length of row_annotation_col.
    col_palette : dict or list of dict, optional
        Single dictionary or list of dictionaries mapping column categories to colors.
        If list, must match length of col_annotation_col.
    figsize : tuple, default (10, 14)
        Figure size (width, height) in inches. Used when creating new figure or
        for font size calculations when ax_size is not provided.
    square : bool, default False
        If True, set heatmap cells to be square-shaped.
    cmap : str or colormap, default 'Blues'
        Colormap for the heatmap. Popular options: 
        - Quantitative: 'viridis', 'plasma', 'RdYlBu_r', 'coolwarm', 'seismic'
        - Qualitative: 'Blues', 'Reds', 'Greens'
    font_scale : float, default 13
        Scale factor for font sizes.
    title : str, default 'Annotated Heatmap'
        Title for the plot.
    xlabel : str, default 'Columns'
        Label for x-axis.
    ylabel : str, default 'Rows'
        Label for y-axis.
    row_patch_width : float or list of float, optional
        Manual width(s) of row annotation patches. If None and auto is True, calculated automatically.
        If list, must match length of row_annotation_col.
    col_patch_height : float or list of float, optional
        Manual height(s) of column annotation patches. If None and auto is True, calculated automatically.
        If list, must match length of col_annotation_col.
    row_patch_auto_width : bool, default True
        Automatically calculate row patch width based on heatmap dimensions.
    col_patch_auto_height : bool, default True
        Automatically calculate column patch height based on heatmap dimensions.
    patch_width_ratio : float, default 0.05
        Ratio of heatmap width for row patches (used when auto width is True).
    patch_height_ratio : float, default 0.05
        Ratio of heatmap height for column patches (used when auto height is True).
    row_annotation_col : str or list of str, optional
        Column name(s) in row_annotations DataFrame to use for coloring.
        If list, creates multiple annotation tracks.
    col_annotation_col : str or list of str, optional
        Column name(s) in col_annotations DataFrame to use for coloring.
        If list, creates multiple annotation tracks.
    row_legend_title : str or list of str, default 'Row Categories'
        Title(s) for row annotation legend(s).
        If list, must match length of row_annotation_col.
    col_legend_title : str or list of str, default 'Column Categories'
        Title(s) for column annotation legend(s).
        If list, must match length of col_annotation_col.
    value_legend_title : str, default 'Values'
        Title for value legend (used for qualitative heatmaps).
    value_legend_labels : dict, optional
        Custom labels for value legend (e.g., {'high': 'Present', 'low': 'Absent'}).
        Only used when heatmap_type='qualitative'.
    legend_position : str, default 'right'
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
        Order of legends. Can contain 'row_0', 'row_1', ..., 'col_0', 'col_1', ..., 'value'.
        Default is all row legends, then all col legends, then value legend.
    save_path : str, optional
        Path to save the figure. If separate_legends is True, individual legend files
        will be saved with '_legend_{key}' appended to the filename.
        If separate_colorbar is True, colorbar will be saved with '_colorbar' appended.
    dpi : int, default 600
        DPI for saved figure.
    show_colorbar : bool, default False
        Whether to show colorbar for heatmap values. Automatically True for quantitative heatmaps.
    colorbar_position : str, default 'right'
        Position of colorbar: 'right', 'left', 'top', or 'bottom'.
        Only used when separate_colorbar=False and colorbar_ax is None.
    colorbar_size : str, default '5%'
        Size of the colorbar (percentage of the axes size).
    colorbar_pad : float, default 0.05
        Padding between heatmap and colorbar.
    colorbar_label : str, optional
        Label for the colorbar.
    cbar_ticks : sequence of float, optional
        Custom tick positions for the colorbar.
    colorbar_orientation : str, optional
        Orientation of colorbar. If None, automatically determined from position.
    colorbar_coords : tuple of 4 floats, optional
        Custom position for colorbar as (left, bottom, width, height) in figure coordinates.
        Overrides automatic positioning. Only used when colorbar_ax is None.
    colorbar_ax : matplotlib.axes.Axes, optional
        Existing axes to use for colorbar. If provided, colorbar will be drawn in this axes
        instead of being created automatically. This allows full control over colorbar positioning.
    separate_colorbar : bool, default False
        If True, create a separate figure for the colorbar instead of attaching it to the main plot.
        Useful for independent positioning in complex layouts.
    colorbar_figsize : tuple of float, optional
        Figure size (width, height) for separate colorbar figure.
        If None, automatically determined based on orientation.
    vmin : float, optional
        Minimum value for colorbar normalization.
    vmax : float, optional
        Maximum value for colorbar normalization.
    center : float, optional
        Value at which to center the colormap (for diverging colormaps).
    robust : bool, default False
        If True, use robust quantiles for colorbar limits.
    heatmap_type : str, default 'qualitative'
        Type of heatmap: 'qualitative' (binary/categorical) or 'quantitative' (gradient).
    linewidths : float, default 0.5
        Width of lines between cells.
    linecolor : str, default 'grey'
        Color of lines between cells.
    row_patch_alpha : float, default 1.0
        Transparency of row annotation patches (0.0-1.0).
    col_patch_alpha : float, default 1.0
        Transparency of column annotation patches (0.0-1.0).
    xticklabels : pd.Series, list, or np.ndarray, optional
        Custom labels for x-axis (columns). If None, uses data.columns.
        Must have the same length as data.columns.
    yticklabels : pd.Series, list, or np.ndarray, optional
        Custom labels for y-axis (rows). If None, uses data.index.
        Must have the same length as data.index.
    xticklabels_rotation : float, default 45
        Rotation angle for x-axis tick labels.
    yticklabels_rotation : float, default 0
        Rotation angle for y-axis tick labels.
    auto_tick_padding : bool, default True
        Automatically calculate tick padding based on patch dimensions.
    tick_pad_x : float, optional
        Manual padding for x-axis ticks. If None and auto is True, calculated automatically.
    tick_pad_y : float, optional
        Manual padding for y-axis ticks. If None and auto is True, calculated automatically.
    tick_pad_ratio : float, default 1.5
        Multiplier for converting patch dimensions to tick padding (in points).
        Higher values create more space between patches and tick labels.
    base_tick_pad : float, default 5.0
        Base padding (in points) added to calculated tick padding.
    font_size_func : callable, optional
        Custom function to calculate font sizes. Should accept (width, height, unit, scale)
        and return a dict with keys: 'ticks_label', 'label', 'legend', 'legend_title', 'title',
        'colorbar_label', 'colorbar_ticks'.
    cbar_kws : dict, optional
        Additional keyword arguments for colorbar customization.
    row_patch_spacing : float, default 0.0
        Spacing between multiple row annotation patches (in data coordinates).
    col_patch_spacing : float, default 0.0
        Spacing between multiple column annotation patches (in data coordinates).
    row_separation_col : str or list of str, optional
        Column name(s) in row_annotations to use for drawing horizontal separation lines
        between different categories. If list, draws multiple levels of separation lines.
    col_separation_col : str or list of str, optional
        Column name(s) in col_annotations to use for drawing vertical separation lines
        between different categories. If list, draws multiple levels of separation lines.
    row_separation_linewidth : float or list of float, default 2.0
        Line width(s) for horizontal separation lines. If list, must match length of row_separation_col.
    col_separation_linewidth : float or list of float, default 2.0
        Line width(s) for vertical separation lines. If list, must match length of col_separation_col.
    row_separation_color : str or list of str, default 'black'
        Color(s) for horizontal separation lines. If list, must match length of row_separation_col.
    col_separation_color : str or list of str, default 'black'
        Color(s) for vertical separation lines. If list, must match length of col_separation_col.
    row_separation_linestyle : str or list of str, default '-'
        Line style(s) for horizontal separation lines. If list, must match length of row_separation_col.
        Options: '-', '--', '-.', ':'
    col_separation_linestyle : str or list of str, default '-'
        Line style(s) for vertical separation lines. If list, must match length of col_separation_col.
        Options: '-', '--', '-.', ':'
    row_separation_alpha : float or list of float, default 1.0
        Alpha (transparency) value(s) for horizontal separation lines (0.0-1.0).
        If list, must match length of row_separation_col. 0.0 is fully transparent, 1.0 is fully opaque.
    col_separation_alpha : float or list of float, default 1.0
        Alpha (transparency) value(s) for vertical separation lines (0.0-1.0).
        If list, must match length of col_separation_col. 0.0 is fully transparent, 1.0 is fully opaque.
    separate_legends : bool, default False
        If True, create separate figure objects for each legend instead of placing them on the main figure.
        Useful for complex layouts or when legends need independent positioning.
    legend_orientation : str, default 'vertical'
        Orientation of legend items: 'vertical' (stacked) or 'horizontal' (side-by-side).
        When 'horizontal', all items in a legend are displayed in one row.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The main figure object containing the heatmap.
    ax : matplotlib.axes.Axes
        The axes object containing the heatmap.
    legend_figs : dict
        Dictionary of legend figure objects if separate_legends=True, otherwise empty dict.
        Keys are legend identifiers (e.g., 'row_0', 'col_0', 'value').
    colorbar_fig : matplotlib.figure.Figure or None
        Separate colorbar figure if separate_colorbar=True, otherwise None.
    
    Examples
    --------
    >>> # Basic usage with automatic colorbar positioning
    >>> fig, ax, legends, cbar_fig = plot_annotated_heatmap(
    ...     data=expression_data,
    ...     heatmap_type='quantitative',
    ...     show_colorbar=True,
    ...     colorbar_label='Expression Level'
    ... )
    
    >>> # Custom colorbar positioning with coords
    >>> fig, ax, legends, cbar_fig = plot_annotated_heatmap(
    ...     data=expression_data,
    ...     show_colorbar=True,
    ...     colorbar_coords=(0.92, 0.3, 0.02, 0.4),  # (left, bottom, width, height)
    ...     colorbar_label='Log2 FC'
    ... )
    
    >>> # Using existing axes with separate colorbar axes
    >>> fig, axes = plt.subplots(1, 2, figsize=(15, 10),
    ...                          gridspec_kw={'width_ratios': [20, 1]})
    >>> plot_annotated_heatmap(
    ...     data=expression_data,
    ...     fig=fig, ax=axes[0],
    ...     show_colorbar=True,
    ...     colorbar_ax=axes[1],
    ...     colorbar_label='Expression'
    ... )
    
    >>> # Separate colorbar figure
    >>> fig, ax, legends, cbar_fig = plot_annotated_heatmap(
    ...     data=expression_data,
    ...     show_colorbar=True,
    ...     separate_colorbar=True,
    ...     colorbar_figsize=(2, 6),
    ...     save_path='heatmap.png'  # Also saves 'heatmap_colorbar.png'
    ... )
    
    >>> # Complex multi-panel layout
    >>> fig = plt.figure(figsize=(20, 10))
    >>> gs = fig.add_gridspec(2, 3, width_ratios=[10, 10, 0.5])
    >>> ax1 = fig.add_subplot(gs[0, 0])
    >>> ax2 = fig.add_subplot(gs[1, 0])
    >>> cbar_ax = fig.add_subplot(gs[:, 2])
    >>> 
    >>> plot_annotated_heatmap(data=data1, fig=fig, ax=ax1, 
    ...                        show_colorbar=True, colorbar_ax=cbar_ax)
    >>> plot_annotated_heatmap(data=data2, fig=fig, ax=ax2,
    ...                        show_colorbar=False)  # Share colorbar
    """
    
    # Validate fig and ax parameters
    if (fig is None) != (ax is None):
        raise ValueError("Both fig and ax must be provided together, or both must be None")
    
    # Determine if we're using existing axes
    use_existing_ax = (fig is not None and ax is not None)

    if transpose:
        # Transpose main data
        data = data.T

        # Swap annotations
        row_annotations, col_annotations = col_annotations, row_annotations
        row_annotation_col, col_annotation_col = col_annotation_col, row_annotation_col
        row_palette, col_palette = col_palette, row_palette

        # Swap labels and titles
        xlabel, ylabel = ylabel, xlabel
        row_legend_title, col_legend_title = col_legend_title, row_legend_title

        # Swap patch dimensions
        row_patch_width, col_patch_height = col_patch_height, row_patch_width
        row_patch_auto_width, col_patch_auto_height = col_patch_auto_height, row_patch_auto_width
        row_patch_spacing, col_patch_spacing = col_patch_spacing, row_patch_spacing
        row_patch_alpha, col_patch_alpha = col_patch_alpha, row_patch_alpha

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

    # [Previous validation and conversion code remains the same...]
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
    
    # [Previous validation code for annotations and separations...]
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
        'annotation': 9,
        'cbar_label': 12,
        'cbar_ticks': 10,
        'label': 10,
        'text': 10,
        'node_label': 6,
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
    
    # Create figure and axis if not provided
    if not use_existing_ax:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare colorbar keyword arguments
    if cbar_kws is None:  
        cbar_kws = {}
    
    # Determine how to handle colorbar
    colorbar_fig = None
    
    if show_colorbar:
        if separate_colorbar:
            # Don't create colorbar with seaborn, we'll make a separate figure
            cbar_enabled = False
        elif colorbar_ax is not None:
            # Use provided axes for colorbar
            cbar_kws.update({'cax': colorbar_ax})
            cbar_enabled = True
        else:
            # Standard colorbar positioning
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
        linewidths=linewidths,
        linecolor=linecolor,
        xticklabels=True,
        yticklabels=True,
        vmin=vmin,
        vmax=vmax,
        center=center,
        robust=robust,
        square=True if square else False
    )
    
    # Handle colorbar customization or creation
    if show_colorbar:
        if cbar_enabled and hasattr(im, 'collections') and len(im.collections) > 0:
            # Customize existing colorbar
            cbar = ax.collections[0].colorbar

            # Customize colorbar position if coords provided and not using separate ax
            if colorbar_coords is not None and colorbar_ax is None:
                cbar.ax.set_position(colorbar_coords)
            
            # Customize colorbar label
            if colorbar_label: 
                cbar.set_label(colorbar_label,
                              fontsize=font_size.get('cbar_label', font_size['label']),
                              rotation=90 if colorbar_orientation == 'vertical' else 0,
                              labelpad=10)
            
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
            
            # Customize colorbar tick labels
            cbar.ax.tick_params(labelsize=font_size.get('cbar_ticks', font_size['ticks_label']))
            
            # Add frame around colorbar
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('black')
        
        elif separate_colorbar:
            # Create separate colorbar figure
            # Determine colorbar figure size
            if colorbar_figsize is None:
                if colorbar_orientation == 'vertical':
                    colorbar_figsize = (2, 6)
                else:
                    colorbar_figsize = (6, 1.5)
            
            colorbar_fig = plt.figure(figsize=colorbar_figsize)
            cbar_ax_separate = colorbar_fig.add_axes([0.1, 0.1, 0.8, 0.8])
            
            # Get normalization from the heatmap
            norm = im.collections[0].norm
            
            # Create colorbar
            cbar = plt.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax_separate,
                orientation=colorbar_orientation
            )
            
            # Customize colorbar
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
            
            # Save separate colorbar if path provided
            if save_path is not None:
                base, ext = save_path.rsplit('.', 1) if '.' in save_path else (save_path, 'png')
                cbar_save_path = f"{base}_colorbar.{ext}"
                colorbar_fig.savefig(cbar_save_path, dpi=dpi, bbox_inches='tight')
    
    # [Rest of the code for tick labels, patches, separation lines, and legends remains the same as before...]
    # Customize tick labels with custom labels if provided
    ax.set_xticklabels(
        xticklabels,
        rotation=xticklabels_rotation,
        ha='center' if xticklabels_rotation == 90 else 'right',
        va='top',
        fontsize=font_size['ticks_label']
    )
    ax.set_yticklabels(
        yticklabels,
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
    
    ax.tick_params(axis='x', which='both', length=0, pad=tick_pad_x)
    ax.tick_params(axis='y', which='both', length=0, pad=tick_pad_y)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    legend_dict = {}
    
    # Add patches for rows
    if row_annotations is not None and row_annotation_col is not None and row_palette is not None:  
        for annot_idx in range(len(row_annotation_col)):
            start_x = xlim[0] - total_row_patch_width + sum(row_patch_width[:annot_idx]) + row_patch_spacing * annot_idx
            col_name = row_annotation_col[annot_idx]
            present_categories = set()
            for i, row_idx in enumerate(data.index):
                category = row_annotations.loc[row_idx, col_name]
                present_categories.add(category)
                color = row_palette[annot_idx].get(category, 'grey')
                rect = Rectangle(
                    (start_x - 0.2, i), row_patch_width[annot_idx], 1,
                    alpha=row_patch_alpha, linewidth=linewidths, edgecolor=linecolor,
                    facecolor=color, clip_on=False, zorder=-1
                )
                ax.add_patch(rect)
            
            legend_elements_row = [
                Rectangle((0, 0), 1, 1, fc=color, ec=linecolor, linewidth=linewidths, label=category)
                for category, color in row_palette[annot_idx].items()
                if category in present_categories
            ]
            legend_dict[f'row_{annot_idx}'] = {
                'handles': legend_elements_row,
                'title': row_legend_title[annot_idx],
                'n_items': len(legend_elements_row)
            }
    
    # Add patches for columns
    if col_annotations is not None and col_annotation_col is not None and col_palette is not None:
        for annot_idx in range(len(col_annotation_col)):
            start_y = ylim[0] + sum(col_patch_height[:annot_idx]) + col_patch_spacing * annot_idx
            col_name = col_annotation_col[annot_idx]
            present_categories = set()
            for j, col_idx in enumerate(data.columns):
                category = col_annotations.loc[col_idx, col_name]
                present_categories.add(category)
                color = col_palette[annot_idx].get(category, 'grey')
                rect = Rectangle(
                    (j, start_y + 0.2), 1, col_patch_height[annot_idx],
                    alpha=col_patch_alpha, linewidth=linewidths, edgecolor=linecolor,
                    facecolor=color, clip_on=False, zorder=-1
                )
                ax.add_patch(rect)
            
            legend_elements_col = [
                Rectangle((0, 0), 1, 1, fc=color, ec=linecolor, linewidth=linewidths, label=category)
                for category, color in col_palette[annot_idx].items()
                if category in present_categories
            ]
            legend_dict[f'col_{annot_idx}'] = {
                'handles': legend_elements_col,
                'title': col_legend_title[annot_idx],
                'n_items': len(legend_elements_col)
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
    
    # Handle legends (same as before)
    if legend_order is None:
        legend_order = []
        if row_annotation_col is not None:
            legend_order.extend([f'row_{i}' for i in range(len(row_annotation_col))])
        if col_annotation_col is not None:
            legend_order.extend([f'col_{i}' for i in range(len(col_annotation_col))])
        legend_order.append('value')
    
    legends_to_plot = [key for key in legend_order if key in legend_dict]
    legend_figs = {}

    def get_ncol(n_items, orientation):
        return n_items if orientation == 'horizontal' else 1

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
    else:
        if legend_auto_spacing and legends_to_plot:
            legend_heights = []
            for key in legends_to_plot:  
                n_items = legend_dict[key]['n_items']
                if legend_orientation == 'horizontal':
                    estimated_height = 0.06 
                else:
                    estimated_height = 0.04 + (n_items * 0.03)
                legend_heights.append(estimated_height)
            
            total_legend_height = sum(legend_heights) + (len(legends_to_plot) - 1) * legend_spacing
            
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

        for idx, key in enumerate(legends_to_plot):
            legend_info = legend_dict[key]
            y_position = legend_positions[idx]
            
            fig.legend(
                handles=legend_info['handles'],
                title=legend_info['title'],
                loc=legend_loc,
                bbox_to_anchor=(bbox_x, y_position),
                ncol=get_ncol(legend_info['n_items'], legend_orientation),
                frameon=True,
                fontsize=font_size.get('legend', 10),
                title_fontsize=font_size.get('legend_title', 12),
            )

    ax.set_title(title, fontsize=font_size.get('title', 14), pad=50)
    
    if not use_existing_ax:
        plt.tight_layout()

    if save_path is not None and not use_existing_ax:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=False)

    return fig, ax, legend_figs, colorbar_fig

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