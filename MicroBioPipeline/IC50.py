import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 4-Parameter Logistic Function
# -------------------------------------------------------------------
def four_param_logistic(x, bottom, top, logIC50, hill_slope):
    """
    4-parameter logistic model:
    
    Y = Bottom + (Top - Bottom) / (1 + 10^((logIC50 - X) * HillSlope))
    """
    return bottom + (top - bottom) / (1 + 10 ** ((logIC50 - x) * hill_slope))



# -------------------------------------------------------------------
# Fit IC50 from a replicates-by-dilutions dataset
# -------------------------------------------------------------------
def fit_ic50(dilutions, responses, confidence=0.95, maxfev=10000):
    """
    Fit IC50 using a 4-parameter logistic regression.

    Parameters
    ----------
    dilutions : array-like (1D)
        Concentrations (not log-transformed). Must match number of rows.
    responses : array-like (2D)
        A matrix with shape (num_dilutions, num_replicates).
        Rows = dilutions, columns = replicates.
    confidence : float
        Confidence interval level (default = 0.95).
    maxfev : int
        Max iterations for curve fitting.

    Returns
    -------
    IC50 : float
    IC50_CI : (lower, upper)
    params : dict {bottom, top, logIC50, hill_slope, covariance}
    """

    dilutions = np.asarray(dilutions)
    responses = np.asarray(responses)

    # Validate dimensions
    if responses.ndim != 2:
        raise ValueError("responses must be a 2D array: rows=dilutions, columns=replicates.")
    if len(dilutions) != responses.shape[0]:
        raise ValueError(
            f"Number of dilutions ({len(dilutions)}) does not match "
            f"number of rows in responses ({responses.shape[0]})."
        )

    # Flatten replicate data
    x = np.log10(np.repeat(dilutions, responses.shape[1]))
    y = responses.flatten()

    # Initial guess for the parameters
    p0 = [y.min(), y.max(), np.median(x), 1.0]

    # Fit model
    params_fit, cov = curve_fit(
        four_param_logistic,
        x,
        y,
        p0=p0,
        maxfev=maxfev
    )

    bottom, top, logIC50, hill_slope = params_fit
    IC50 = 10 ** logIC50

    # ---- 95% CI for IC50 ----
    alpha = 1 - confidence
    dof = len(x) - len(params_fit)
    tval = t.ppf(1 - alpha / 2, dof)

    se_logIC50 = np.sqrt(cov[2, 2])
    log_low = logIC50 - tval * se_logIC50
    log_high = logIC50 + tval * se_logIC50

    IC50_CI = (10 ** log_low, 10 ** log_high)

    params = {
        "bottom": bottom,
        "top": top,
        "logIC50": logIC50,
        "hill_slope": hill_slope,
        "covariance": cov
    }

    return IC50, IC50_CI, params



# -------------------------------------------------------------------
# Plot the fitted IC50 curve with replicate data
# -------------------------------------------------------------------
def plot_ic50(dilutions, responses, params, IC50=None, IC50_CI=None,
              save_path=None, dpi=300, show=True):
    """
    Plot dose-response curve with mean ± SD for replicates.
    """

    dilutions = np.asarray(dilutions, dtype=float)
    responses = np.asarray(responses, dtype=float)

    # --------------------------------
    # Compute mean and standard deviation across replicates
    # --------------------------------
    means = responses.mean(axis=1)     # shape (N,)
    sds = responses.std(axis=1)        # shape (N,)

    # Convert to log10 x-values
    x = np.log10(dilutions)

    # --------------------------------
    # Get fitted parameters
    # --------------------------------
    bottom = params["bottom"]
    top = params["top"]
    logIC50 = params["logIC50"]
    hill_slope = params["hill_slope"]

    # Smooth fit curve
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = four_param_logistic(x_fit, bottom, top, logIC50, hill_slope)

    # --------------------------------
    # Plot
    # --------------------------------
    plt.figure(figsize=(7, 5))
    font_dict = get_font_sizes(7, 5, unit="in")

    # Mean ± SD points
    plt.errorbar(
        x, means, yerr=sds,
        fmt='o', capsize=5, label="Mean ± SD", linestyle='none'
    )

    # Fitted curve
    plt.plot(x_fit, y_fit, linewidth=2, label="4PL Fit")

    plt.xlabel("log10(Inhibitor concentration)", fontsize=font_dict['axes_label'])
    plt.ylabel("Response", fontsize=font_dict['axes_label'])

    if IC50 is not None and IC50_CI is not None:
        title_text = (f"4-Parameter Logistic IC50 Fit\n"
                      f"IC50: {IC50:.2e}  CI: [{IC50_CI[0]:.2e}, {IC50_CI[1]:.2e}]")
    else:
        title_text = "4-Parameter Logistic IC50 Fit"

    plt.title(title_text, fontsize=font_dict['title'])
    plt.legend(fontsize=font_dict['legend'])
    plt.tight_layout()

    # Save if needed
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()



def plot_ic50_layers(layers, params_list, IC50_list=None, IC50_CI_list=None, dilutions=None,
                     labels=None, save_path=None, dpi=300, show=True):
    """
    Plot dose-response curves for multiple layers with mean ± SD for replicates.

    Parameters
    ----------
    layers : list of np.ndarray
        Each element is a 2D array (replicates x dilutions) for one layer.
    params_list : list of dict
        Fitted parameters for each layer: bottom, top, logIC50, hill_slope.
    IC50_list : list, optional
        List of IC50 values for each layer.
    IC50_CI_list : list of tuples, optional
        List of confidence intervals (low, high) for each IC50.
    labels : list of str, optional
        Labels for each layer for the legend.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    font_dict = get_font_sizes(7, 5, unit="in")

    if labels is None:
        labels = [f"Layer {i+1}" for i in range(len(layers))]

    colors = plt.cm.tab10.colors  # Up to 10 colors
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    for i, layer in enumerate(layers):
        layer = np.asarray(layer, dtype=float)

        # Compute mean ± SD across replicates
        means = layer.mean(axis=1)  # assuming shape (replicates, dilutions)
        sds = layer.std(axis=1)

        x = np.log10(dilutions.astype(float))  # If dilutions are repeated across replicates

        # Fitted curve
        bottom = params_list[i]["bottom"]
        top = params_list[i]["top"]
        logIC50 = params_list[i]["logIC50"]
        hill_slope = params_list[i]["hill_slope"]

        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = four_param_logistic(x_fit, bottom, top, logIC50, hill_slope)

        # Plot mean ± SD
        # plt.errorbar(
        #     x, means, yerr=sds,
        #     fmt='o', capsize=5, label=f"{labels[i]} Mean ± SD",
        #     color=colors[i % len(colors)], linestyle='none'
        # )

        plt.scatter(
            x, means,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=80,
            label='_nolegend_'
        )

        # Plot fit
        plt.plot(
            x_fit, y_fit,
            linewidth=2,
            color=colors[i % len(colors)],
            label=f"{labels[i]}"
        )

        # Optionally add IC50 in legend
        if IC50_list is not None and IC50_CI_list is not None:
            ic50_text = f"IC50: {IC50_list[i]:.2e}  CI: [{IC50_CI_list[i][0]:.2e}, {IC50_CI_list[i][1]:.2e}]"
            plt.text(0.05, 0.1 - i*0.05, f"{labels[i]}: {ic50_text}", transform=plt.gca().transAxes, fontsize=font_dict['legend'], verticalalignment='bottom')

    plt.xlabel("log10(Inhibitor concentration)", fontsize=font_dict['axes_label'])
    plt.ylabel("Response", fontsize=font_dict['axes_label'])
    plt.title("4-Parameter Logistic IC50 Fit", fontsize=font_dict['title'])
    plt.legend(fontsize=font_dict['legend'])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

