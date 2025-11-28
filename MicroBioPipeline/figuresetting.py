import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys



# --------------------------------------------------------------------------------------------------------------
# Define function to suggest font sizes based on figure dimensions
def get_font_sizes(width, height, unit="in"):
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
    base_scale = (width * height) ** 0.5 / 5  # empirically tuned

    font_sizes = {
        'title': round(14 * base_scale),
        'suptitle': round(16 * base_scale),
        'axes_label': round(10 * base_scale),
        'ticks_label': round(10 * base_scale),
        'legend': round(10 * base_scale),
        'legend_title': round(11 * base_scale),
        'annotation': round(9 * base_scale),
        'cbar_label': round(12 * base_scale),
        'cbar_ticks': round(10 * base_scale),
    }

    return font_sizes

# --------------------------------------------------------------------------------------------------------------
# Define function to create pastel colormap
def pastelize_cmap(cmapname='viridis', N=256, sat_scale=0.1, light_offset=0.9, name=None):
    """
    Make a pastel version of a colormap by scaling saturation and shifting lightness.

    Parameters:
      cmapname   : name or Colormap object (default 'viridis')
      N          : number of samples to build the new cmap
      sat_scale  : multiply saturation by this (0..1 for more pastel)
      light_offset: add this to lightness (0..1)
      name       : name for the new colormap
    Returns:
      matplotlib.colors.LinearSegmentedColormap
    """
    if name is None:
        name = f"{cmapname}_pastel"
    base = plt.get_cmap(cmapname)
    colors = base(np.linspace(0, 1, N))[:, :3]  # drop alpha if present

    pastel_colors = []
    for r, g, b in colors:
        # convert rgb -> hls, adjust, convert back
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s = max(0, min(1, s * sat_scale))
        l = max(0, min(1, l + light_offset))
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        pastel_colors.append((r2, g2, b2))

    return LinearSegmentedColormap.from_list(name, pastel_colors, N=N)