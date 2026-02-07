"""
Module: figuresetting.py
This module provides utilities for setting figure aesthetics in matplotlib, including
functions to suggest font sizes based on figure dimensions and to create pastelized colormaps.

Functions:
- get_font_sizes: Suggest font sizes for various plot elements based on figure size.
- pastelize_cmap: Create a pastel version of a given colormap by adjusting saturation and lightness.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys



# --------------------------------------------------------------------------------------------------------------
# Define function to suggest font sizes based on figure dimensions
def get_font_sizes(width, height, unit="in", scale=5.0, type="text"):
    """
    Suggest font sizes for a matplotlib plot based on figure size.

    Parameters
    ----------
    width : float
        Width of the figure (in inches or millimeters).
    height : float
        Height of the figure (in inches or millimeters).
    unit : str, optional
        Unit of input dimensions: "in" for inches (default) or "mm" for millimeters.

    Returns
    -------
    dict
        Dictionary of recommended font sizes for consistent readability.
    """

    # Convert mm to inches if necessary
    if unit.lower() == "mm":
        width /= 25.4
        height /= 25.4
    elif unit.lower() != "in":
        raise ValueError("Unit must be 'in' or 'mm'")

    # Scale factor proportional to sqrt(area)
    base_scale = (width * height) ** 0.5 / scale  # empirically tuned

    font_sizes = {
        'title': round(14 * base_scale),
        'suptitle': round(16 * base_scale),
        'axes_label': round(10 * base_scale),
        'ticks_label': round(5 * base_scale) if type == "text" else round(10 * base_scale),
        'legend': round(7 * base_scale),
        'legend_title': round(8 * base_scale),
        'annotation': round(4 * base_scale),
        'cbar_label': round(12 * base_scale),
        'cbar_ticks': round(10 * base_scale),
        'label': round(10 * base_scale),
        'text': round(10 * base_scale),
        'node_label': round(6 * base_scale),
        'edge_label': round(6 * base_scale),
    }

    return font_sizes

# --------------------------------------------------------------------------------------------------------------
# Define function to create pastel colormap
from typing import Optional, Union, Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import colorsys

def pastelize_cmap(cmapname: Union[str, mcolors.Colormap, Iterable] = 'viridis',
                   N: int = 256,
                   sat_scale: float = 0.1,
                   light_offset: float = 0.9,
                   name: Optional[str] = None,
                   return_array: bool = False):
    """
    Make a pastel version of a colormap by scaling saturation and shifting lightness.

    Parameters:
      cmapname     : name or Colormap object (default 'viridis') OR an iterable/array of colors.
                     If given an iterable/array it may be:
                        - Nx3 or Nx4 numeric array of floats in [0,1]
                        - iterable of matplotlib color specs (hex strings, color names, etc.)
      N            : number of samples to build the new cmap (or length of returned array)
      sat_scale    : multiply saturation by this (0..1 for more pastel)
      light_offset : add this to lightness (0..1; clamped)
      name         : name for the new colormap. If None, a default is constructed.
      return_array : if True, return an (N,3) numpy array of RGB colors instead of a Colormap

    Returns:
      LinearSegmentedColormap by default, or an (N,3) ndarray if return_array is True.
    """
    # clamp helper
    def _clamp(x):
        return max(0.0, min(1.0, float(x)))

    # Determine whether cmapname is a colormap identifier/object or a color vector/iterable.
    is_colormap_like = False
    if isinstance(cmapname, str):
        is_colormap_like = True
    elif isinstance(cmapname, mcolors.Colormap):
        is_colormap_like = True
    else:
        # Explicitly detect numpy arrays and other iterables that are NOT strings.
        if isinstance(cmapname, np.ndarray):
            # Numeric 2D arrays should be treated as color vectors
            if cmapname.ndim == 0:
                # scalar numpy value -> treat as colormap name attempt fallback
                is_colormap_like = isinstance(cmapname.item(), str)
            else:
                is_colormap_like = False
        else:
            # If it's not a numpy array and has __iter__ (but is not a string), assume it's a color vector.
            is_colormap_like = False

    base_colors = None
    if is_colormap_like:
        # Safe: only call get_cmap when we know cmapname is a string or a Colormap instance.
        base = plt.get_cmap(cmapname)
        sampled = base(np.linspace(0.0, 1.0, N))
        base_colors = np.asarray(sampled)[:, :3]
        if name is None:
            try:
                name = base.name
            except Exception:
                name = f"{str(cmapname)}_pastel"
            name = f"{name}_pastel" if not name.endswith("_pastel") else name
    else:
        # Treat as a sequence/array of colors.
        arr = np.asarray(cmapname, dtype=object)

        # If it's a numeric 2D array with shape (M,3) or (M,4)
        if arr.ndim == 2 and arr.shape[1] in (3, 4) and np.issubdtype(arr.dtype, np.number):
            base_colors = np.asarray(arr, dtype=float)[:, :3]
        else:
            # Otherwise treat as sequence of color specifications (strings, tuples, etc.)
            try:
                seq = list(cmapname)
            except Exception:
                raise ValueError("cmapname must be a colormap name/object or an iterable/array of colors")
            # Convert each element to RGB using matplotlib's to_rgb
            rgb_list = [mcolors.to_rgb(c) for c in seq]
            base_colors = np.asarray(rgb_list, dtype=float)[:, :3]
        if name is None:
            name = "custom_pastel"

    # Pastelize by converting rgb -> hls, scaling s, shifting l, converting back
    pastel_colors = []
    for (r, g, b) in base_colors:
        h, l, s = colorsys.rgb_to_hls(_clamp(r), _clamp(g), _clamp(b))
        s_new = _clamp(s * float(sat_scale))
        l_new = _clamp(l + float(light_offset))
        r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s_new)
        pastel_colors.append((r2, g2, b2))
    pastel_colors = np.asarray(pastel_colors, dtype=float)

    # If the caller wants an array of length N, interpolate (or sample) the pastel colors to length N.
    if return_array:
        if len(pastel_colors) == N:
            return pastel_colors
        # Use LinearSegmentedColormap.from_list to get smooth interpolation, then sample it.
        cmap_out = LinearSegmentedColormap.from_list(name, pastel_colors, N=max(len(pastel_colors), N))
        return cmap_out(np.linspace(0, 1, N))[:, :3]

    # Otherwise return a Colormap object (interpolated to N)
    return LinearSegmentedColormap.from_list(name, pastel_colors, N=N)


# ----------------------------------------------------------------------------------------------
# Associate to each label a color
def get_color_dict(labels):
    n_lab = len(labels)
    cmap = plt.cm.tab20
    cmapB = plt.cm.tab20b
    cmapC = plt.cm.tab20c

    colors = np.vstack([
        cmap(np.linspace(0,1,20)),
        cmapB(np.linspace(0,1,20)),
        cmapC(np.linspace(0,1,20)),
        plt.cm.hsv(np.linspace(0,1,n_lab-60 if n_lab > 60 else 0))  # fill to reach 200
    ])

    colors = pastelize_cmap(colors, N=256, sat_scale=0.9, light_offset=0.2, return_array=True)

    color_dict = {}
    for i, label in enumerate(labels):
        # HEX
        import matplotlib as mpl
        color_dict[label] = mpl.colors.rgb2hex(colors[i])
        
    return color_dict