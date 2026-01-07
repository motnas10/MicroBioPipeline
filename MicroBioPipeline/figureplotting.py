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
    title="Dimensionality Reduction",
    font_sizes=None,
    legend_title=None,
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
):
    """
    General-purpose dimensionality reduction plot for PCA/PCoA/tSNE/UMAP/etc.

    - coords_df: dataframe containing embedding results
    - dr_method: 'PCA', 'PCoA', 'tSNE', 'UMAP', or 'auto' (tries to infer from col names, for axis labeling)
    - coords_cols: list of columns to use for coordinates; must be length 2, e.g. ['PC1','PC2'] or ['UMAP1','UMAP2']
    - explained: list of explained variances for axes (optional, mainly for PCA/PCoA)
    """
    if coords_cols is None:
        # Try to infer coordinates: prefer first two columns matching known DR axes
        candidates = ['PC1','PC2','PCoA1','PCoA2','Dim1','Dim2','UMAP1','UMAP2','tSNE1','tSNE2']
        found = [c for c in candidates if c in coords_df.columns]
        if len(found) >= 2:
            coords_cols = found[:2]
        else:
            coords_cols = coords_df.columns[:2].tolist()
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

    fig, ax = plt.subplots(figsize=(10, 10))

    # Style and color mappings
    marker_symbols = ['o', 'X', '^', 's', 'D', 'v', 'P', '*', '<', '>', '8', 'p', 'H', 'h', '+', 'x']
    style_map = {}
    if style_col is not None:
        unique_styles = coords_df[style_col].dropna().unique()
        style_map = {val: marker_symbols[i % len(marker_symbols)] for i, val in enumerate(unique_styles)}
        coords_df['_marker_'] = coords_df[style_col].map(style_map)
    else:
        coords_df['_marker_'] = 'o'
    # Color palette
    unique_colors = coords_df[color_col].dropna().unique() if color_col else ["__single__"]
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
    ax.set_xlabel(x_label, fontsize=font_sizes.get('axes_label',12) if font_sizes else 12)
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
        palette_ellipse = sns.color_palette('Set2', n_colors=len(group_labels))
        group_color_map = dict(zip(group_labels, palette_ellipse))
        for group_name, group_data in coords_df.groupby('_group_'):
            x_vals = group_data[x].values
            y_vals = group_data[y].values
            centroid_x = np.mean(x_vals)
            centroid_y = np.mean(y_vals)
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
                fontsize=font_sizes.get('loadings',12) if font_sizes else 11, 
                ha='center', 
                va='center'
            )

    # ---- Legend ----
    legend_handles = []
    legend_labels = []
    # Color legend
    if color_col is not None:
        legend_handles.append(Line2D([0], [0], color='none'))
        legend_labels.append('Color Style')
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
    # Ellipse legend
    if ellipse_legend_elements:
        legend_handles.append(Line2D([0], [0], color='none'))
        legend_labels.append('Ellipsoid')
        for lab, handle in sorted(ellipse_legend_elements, key=lambda x: x[0]):
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
    if filepath:
        plt.savefig(filepath, dpi=600, bbox_inches='tight', transparent=False)
    plt.show()


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