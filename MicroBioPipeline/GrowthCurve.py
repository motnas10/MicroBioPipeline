"""
GrowthCurve.py
~~~~~~~~~~~~~~
A Python reimplementation of the R package ``growthcurver``
(https://github.com/sprouffske/growthcurver), incorporating all features
described in the official CRAN vignette:
https://cran.r-project.org/web/packages/growthcurver/vignettes/Growthcurver-vignette.html

Overview
--------
Fits growth curve data to the standard logistic equation used in ecology
and evolution:

    N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))

where:
    K   = carrying capacity (maximum sustainable population size)
    N0  = initial population size (absorbance / cell count at t=0)
    r   = intrinsic growth rate (unrestricted exponential growth rate)
    t   = time

Derived metrics (also returned):
    t_mid   = time at the inflection point (N = K/2), marks the transition
              from accelerating to decelerating growth
    t_gen   = maximum doubling time = log(2) / r
    auc_l   = area under the logistic curve (theoretical)
    auc_e   = empirical area under the curve (trapezoidal, from raw data)
    sigma   = residual standard error of the fit (smaller = better fit)

Public API
----------
Core math:
    n_at_t(k, n0, r, t)
    slope_at_t(k, n0, r, t)
    max_doubling_time(r)
    doubling_time_at_t(k, n0, r, t)
    t_at_inflection(k, n0, r)
    area_under_curve(k, n0, r, t_start, t_end)
    empirical_auc(data_t, data_n, t_max)

Main analysis:
    summarize_growth(data_t, data_n, t_trim, bg_correct, blank) -> GrowthFit
    summarize_growth_by_plate(plate_df, t_trim, bg_correct)     -> pd.DataFrame

Quality control:
    quality_control(plate_summary_df, sigma_threshold)          -> pd.DataFrame
    plot_sigma_histogram(plate_summary_df)

Dependencies
------------
    numpy, scipy, pandas, matplotlib


Usage
------------
import numpy as np
import pandas as pd
from GrowthCurve import (
    n_at_t,
    summarize_growth,
    summarize_growth_by_plate,
    quality_control,
    plot_sigma_histogram,
    plot_plate,
)

# ── 1. Single well (vignette: "A simple first example") ─────────────────────
k_in, n0_in, r_in = 0.5, 1e-5, 1.2
data_t = np.linspace(0, 24, 51)
data_n = n_at_t(k_in, n0_in, r_in, data_t)

gc = summarize_growth(data_t, data_n)
gc.summary()          # prints all metrics
gc.plot(title="A1")   # raw data + fitted curve + t_mid line

# ── 2. Full plate (vignette: "Get growth curves for a plate") ────────────────
# Build a synthetic 96-well plate
rng = np.random.default_rng(42)
time_col = data_t
plate = {"time": time_col}
for well in [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)]:
    k_w  = rng.uniform(0.3, 0.7)
    r_w  = rng.uniform(0.8, 1.5)
    n0_w = rng.uniform(5e-6, 2e-5)
    plate[well] = n_at_t(k_w, n0_w, r_w, time_col) + rng.normal(0, 0.002, len(time_col))

plate_df = pd.DataFrame(plate)

summary = summarize_growth_by_plate(plate_df, t_trim=20, bg_correct="min")
print(summary.head())

# ── 3. Quality control (vignette: "Quality control and best practices") ──────
qc = quality_control(summary)

# Show only flagged wells
flagged = qc[qc["flagged"]]
print(f"\nFlagged wells ({len(flagged)}):")
print(flagged[["sample", "sigma", "note", "qc_reason"]])

# Plot sigma histogram to spot outliers visually
plot_sigma_histogram(summary)

# ── 4. Plot all wells in a grid (vignette plate plot) ───────────────────────
plot_plate(plate_df, plate_summary=summary, t_trim=20)

"""

import warnings
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# 1. Core logistic math functions
# ---------------------------------------------------------------------------

def n_at_t(
    k: float,
    n0: float,
    r: float,
    t: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Return population size N at time t under the logistic model.

    Evaluates:
        N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))

    Parameters
    ----------
    k:  Carrying capacity — the maximum sustainable population size.
    n0: Initial population size (absorbance or cell count at t=0).
    r:  Intrinsic growth rate — the rate if there were no resource limits.
    t:  Time point(s) at which to evaluate N.

    Returns
    -------
    N(t) as a float or numpy array.

    Examples
    --------
    >>> n_at_t(k=0.5, n0=1e-5, r=1.2, t=0)
    1e-05
    """
    return k / (1 + ((k - n0) / n0) * np.exp(-r * t))


def slope_at_t(
    k: float,
    n0: float,
    r: float,
    t: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Return the instantaneous growth rate (dN/dt) at time t.

    Parameters
    ----------
    k, n0, r: Logistic parameters.
    t:        Time point(s).

    Returns
    -------
    dN/dt at time t — the slope of the growth curve.
    """
    n = n_at_t(k, n0, r, t)
    return r * n * (k - n) / k


def max_doubling_time(r: float) -> float:
    """Return the minimum (fastest) doubling time assuming exponential growth.

    This is the doubling time when there are no resource limitations, i.e.,
    during the early exponential phase. Computed as log(2) / r.

    Parameters
    ----------
    r: Intrinsic growth rate.

    Returns
    -------
    Minimum doubling time (same time units as the input data).

    Examples
    --------
    >>> round(max_doubling_time(1.2), 4)
    0.5776
    """
    return math.log(2) / r


def doubling_time_at_t(
    k: float,
    n0: float,
    r: float,
    t: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Return the local doubling time at time t under the logistic model.

    As the population approaches K, the doubling time increases because
    growth slows. This function returns that local value.

    Parameters
    ----------
    k, n0, r: Logistic parameters.
    t:        Time point(s).

    Returns
    -------
    Local doubling time(s) at the given time point(s).
    """
    n_t = n_at_t(k, n0, r, t)
    n_half = 0.5 * n_t
    return (1 / r) * np.log((n_t * (k - n_half)) / ((k - n_t) * n_half))


def t_at_inflection(k: float, n0: float, r: float) -> float:
    """Return the time at the inflection point of the logistic curve (N = K/2).

    The inflection point (called ``t_mid`` in outputs) is where growth
    transitions from accelerating to decelerating — i.e., where dN/dt
    is maximal. It occurs when N = K/2.

    Parameters
    ----------
    k:  Carrying capacity.
    n0: Initial population size.
    r:  Growth rate.

    Returns
    -------
    Time of the inflection point.

    Warns
    -----
    UserWarning if n0 == 0 (undefined result).
    """
    if n0 == 0:
        warnings.warn("Initial population size (n0) cannot be 0.")
        return 0.0
    return math.log(abs(k - n0) / n0) / r


def area_under_curve(
    k: float,
    n0: float,
    r: float,
    t_start: float,
    t_end: float,
) -> float:
    """Return the area under the logistic curve from t_start to t_end (auc_l).

    ``auc_l`` integrates the effects of N0, K, and r into a single fitness
    proxy. It represents the theoretical total growth under the fitted model.
    Useful for comparing curves even when K or r differ.

    Uses scipy.integrate.quad for numerical integration.

    Parameters
    ----------
    k, n0, r: Logistic parameters.
    t_start:  Lower bound of integration (usually 0).
    t_end:    Upper bound of integration (usually the last time point).

    Returns
    -------
    Numeric area under the logistic curve (auc_l).
    """
    result, _ = quad(lambda t: n_at_t(k, n0, r, t), t_start, t_end)
    return result


def empirical_auc(
    data_t: np.ndarray,
    data_n: np.ndarray,
    t_max: float,
) -> float:
    """Return the empirical area under the curve using the trapezoidal rule (auc_e).

    ``auc_e`` is computed directly from the raw observed data, without relying
    on the fitted logistic model. Comparing ``auc_l`` and ``auc_e`` is a useful
    quality check: large differences suggest a poor model fit.

    Parameters
    ----------
    data_t: Array of time points.
    data_n: Array of corresponding population measurements.
    t_max:  Only data up to and including this time are used.

    Returns
    -------
    Trapezoidal area under the empirical growth curve (auc_e).
    """
    mask = data_t <= t_max
    return float(np.trapz(data_n[mask], data_t[mask]))


# ---------------------------------------------------------------------------
# 2. Result containers
# ---------------------------------------------------------------------------

@dataclass
class GrowthVals:
    """All fitted parameters and derived metrics from a single growth curve.

    These mirror the fields in growthcurver's ``gcvals`` R object.

    Parameters / Attributes
    -----------------------
    k, k_se, k_p:
        Carrying capacity, its standard error, and two-tailed p-value.
        K is the maximum sustainable population size.
    n0, n0_se, n0_p:
        Initial population size (at t=0), its SE, and p-value.
    r, r_se, r_p:
        Intrinsic growth rate, its SE, and p-value. Higher r = faster growth.
    sigma:
        Residual standard error of the logistic fit. Smaller values indicate
        a better fit. Use this to identify outlier wells in a plate experiment.
    df:
        Degrees of freedom from the nonlinear regression (n_obs - 3).
    t_mid:
        Time at the inflection point (N = K/2). Marks the transition from
        exponential to decelerating growth. Should be positive; if negative,
        the fit is flagged as questionable.
    t_gen:
        Maximum (fastest) doubling time = log(2) / r. Smaller t_gen means
        faster growth. Equivalent to generation time in exponential phase.
    auc_l:
        Area under the fitted logistic curve. A composite fitness proxy that
        integrates K, N0, and r. Useful for comparing samples.
    auc_e:
        Empirical (trapezoidal) area under the raw data curve. Compare with
        auc_l: large discrepancy suggests a poor fit.
    note:
        A string warning about common fit problems:
        - ""                          : fit looks good
        - "cannot fit data"           : the optimizer failed entirely
        - "questionable fit (k < n0)" : carrying capacity < initial size
        - "questionable fit"          : t_mid is negative
    """
    k: float = 0.0
    k_se: float = 0.0
    k_p: float = 0.0
    n0: float = 0.0
    n0_se: float = 0.0
    n0_p: float = 0.0
    r: float = 0.0
    r_se: float = 0.0
    r_p: float = 0.0
    sigma: float = 0.0
    df: int = 0
    t_mid: float = 0.0
    t_gen: float = 0.0
    auc_l: float = 0.0
    auc_e: float = 0.0
    note: str = ""


@dataclass
class GrowthFit:
    """Complete result of fitting a logistic model to a single growth curve.

    Mirrors growthcurver's ``gcfit`` R object.

    Attributes
    ----------
    vals:   A :class:`GrowthVals` instance with all parameters and metrics.
    model:  Dict with keys ``popt`` ([k, n0, r]), ``pcov``, ``df``
            from scipy curve_fit, or ``None`` if fitting failed.
    data:   Dict with keys ``'t'`` and ``'N'`` holding the
            (background-corrected) input arrays used for fitting.

    Methods
    -------
    plot(ax=None):
        Plot the raw data and the fitted logistic curve.
    summary():
        Print a human-readable summary of the fit.
    """
    vals: GrowthVals = field(default_factory=GrowthVals)
    model: Optional[dict] = None
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return self.summary(print_output=False)

    def summary(self, print_output: bool = True) -> str:
        """Return (and optionally print) a human-readable summary of the fit.

        Parameters
        ----------
        print_output: If True (default), also prints the summary to stdout.

        Returns
        -------
        Formatted string summary.
        """
        v = self.vals
        lines = [
            "Fit: K / (1 + ((K - N0) / N0) * exp(-r * t))",
            f"  K   = {v.k:.4f}  (SE={v.k_se:.4g},  p={v.k_p:.4g})",
            f"  N0  = {v.n0:.4g}  (SE={v.n0_se:.4g},  p={v.n0_p:.4g})",
            f"  r   = {v.r:.4f}  (SE={v.r_se:.4g},  p={v.r_p:.4g})",
            f"  sigma={v.sigma:.4g},  df={v.df}",
            "",
            "Derived metrics:",
            f"  t_mid = {v.t_mid:.4f}  (inflection point, N = K/2)",
            f"  t_gen = {v.t_gen:.4f}  (max doubling time = log(2)/r)",
            f"  auc_l = {v.auc_l:.4f}  (area under logistic curve)",
            f"  auc_e = {v.auc_e:.4f}  (empirical trapezoidal AUC)",
        ]
        if v.note:
            lines.append(f"\n  ⚠ Note: {v.note}")
        result = "\n".join(lines)
        if print_output:
            print(result)
        return result

    def plot(self, ax=None, title: str = "", show: bool = True):
        """Plot raw data points and the fitted logistic curve.

        Mirrors the ``plot(gc_fit)`` call in the R vignette.

        Parameters
        ----------
        ax:    A matplotlib Axes object. If None, a new figure is created.
        title: Optional title string for the plot.
        show:  If True (default), calls plt.show() after plotting.

        Returns
        -------
        The matplotlib Axes object.

        Raises
        ------
        ImportError if matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            ) from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        t_data = self.data.get("t", np.array([]))
        n_data = self.data.get("N", np.array([]))

        # Raw data
        ax.scatter(t_data, n_data, s=15, color="steelblue",
                   label="Observed", zorder=3)

        # Fitted logistic curve
        v = self.vals
        if self.model is not None and v.r > 0:
            t_fit = np.linspace(float(np.min(t_data)), float(np.max(t_data)), 300)
            n_fit = n_at_t(v.k, v.n0, v.r, t_fit)
            ax.plot(t_fit, n_fit, color="firebrick", linewidth=1.5,
                    label="Logistic fit")
            # Mark the inflection point (t_mid)
            ax.axvline(v.t_mid, color="gray", linestyle="--", linewidth=0.8,
                       label=f"t_mid={v.t_mid:.2f}")

        ax.set_xlabel("Time")
        ax.set_ylabel("Population (N)")
        ax.set_title(title or "Growth curve fit")
        ax.legend(fontsize=8)

        if v.note:
            ax.text(0.02, 0.95, f"⚠ {v.note}", transform=ax.transAxes,
                    fontsize=7, color="red", va="top")

        if show:
            plt.tight_layout()
            plt.show()

        return ax


# ---------------------------------------------------------------------------
# 3. Internal fitting helpers
# ---------------------------------------------------------------------------

def _logistic(t: np.ndarray, k: float, n0: float, r: float) -> np.ndarray:
    """Logistic function for use with scipy.optimize.curve_fit (internal)."""
    return k / (1 + ((k - n0) / n0) * np.exp(-r * t))


def _fit_logistic(
    data_t: np.ndarray,
    data_n: np.ndarray,
) -> Optional[dict]:
    """Fit the logistic model; returns a result dict or None on failure.

    Mirrors the R ``FitLogistic()`` function, including the same initial-guess
    heuristics (logit-linear regression for r, max for K, min for N0).

    Parameters
    ----------
    data_t: 1-D array of time points.
    data_n: 1-D array of population measurements (background-corrected).

    Returns
    -------
    Dict with keys ``popt`` ([k, n0, r]), ``pcov``, and ``df``,
    or ``None`` if the optimisation fails.
    """
    data_t = np.asarray(data_t, dtype=float)
    data_n = np.asarray(data_n, dtype=float)

    # --- Initial guesses (matching growthcurver's heuristics) ---
    k_init = float(np.max(data_n))
    positive = data_n[data_n > 0]
    n0_init = float(np.min(positive)) if len(positive) > 0 else 1e-6

    # Estimate r via linear regression of logit(n / k_init)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(data_n / k_init, 1e-9, 1 - 1e-9)
        logit_vals = np.log(ratio / (1 - ratio))
    finite_mask = np.isfinite(logit_vals)
    if finite_mask.sum() >= 2:
        slope = np.polyfit(data_t[finite_mask], logit_vals[finite_mask], 1)[0]
        r_init = max(float(slope), 0.001)
    else:
        r_init = 0.001

    try:
        popt, pcov = curve_fit(
            _logistic,
            data_t,
            data_n,
            p0=[k_init, n0_init, r_init],
            bounds=(
                [float(np.median(data_n)), 0.0, 0.0],
                [np.inf, float(np.max(data_n)), np.inf],
            ),
            maxfev=5000,
        )
        df = max(len(data_t) - 3, 1)  # 3 parameters: k, n0, r
        return {"popt": popt, "pcov": pcov, "df": df}
    except Exception:
        return None


def _compute_p_value(estimate: float, se: float, df: int) -> float:
    """Two-tailed p-value for H₀: parameter = 0 (internal)."""
    if se == 0 or df <= 0:
        return float("nan")
    t_stat = estimate / se
    return float(2 * t_dist.sf(abs(t_stat), df))


# ---------------------------------------------------------------------------
# 4. Main public functions
# ---------------------------------------------------------------------------

def summarize_growth(
    data_t: Union[list, np.ndarray],
    data_n: Union[list, np.ndarray],
    t_trim: float = 0,
    bg_correct: str = "min",
    blank: Optional[Union[list, np.ndarray]] = None,
) -> GrowthFit:
    """Fit a logistic growth model to a single growth curve.

    This is the primary analysis function. It mirrors R's ``SummarizeGrowth()``.

    Background correction
    ---------------------
    Per the vignette, background correction is important. Without it,
    Growthcurver may fail to fit data properly because N0 will not be
    near zero. Three methods are supported:

    * ``"min"``   (default): subtract the minimum value in the data column
                             from all rows (simple and robust).
    * ``"blank"``:           subtract a blank (media-only) well reading from
                             each corresponding time point.
    * ``"none"``:            no correction (use only if you pre-corrected data).

    Time trimming
    -------------
    If the culture did not reach stationary phase (or if stationary phase
    is variable), fitting can be improved by trimming data after ``t_trim``
    hours. Set ``t_trim=0`` to use all data (default).

    Parameters
    ----------
    data_t:     1-D array of time points (e.g., in hours).
    data_n:     1-D array of absorbance / cell-count readings, same length.
    t_trim:     Discard measurements taken after this time. 0 = no trimming.
    bg_correct: Background correction method: ``"min"``, ``"blank"``,
                or ``"none"``.
    blank:      1-D array of blank-well readings, required only when
                ``bg_correct="blank"``.

    Returns
    -------
    :class:`GrowthFit` containing:
        - ``vals``  : :class:`GrowthVals` with K, N0, r, t_mid, t_gen,
                      auc_l, auc_e, sigma, df, and quality notes.
        - ``model`` : raw scipy curve_fit result (popt, pcov, df).
        - ``data``  : the background-corrected (t, N) arrays used for fitting.

    Quality notes (vals.note)
    -------------------------
    * ``"cannot fit data"``          : optimizer failed entirely.
    * ``"questionable fit (k < n0)"``: K is smaller than N0 — unexpected.
    * ``"questionable fit"``         : t_mid is negative — unexpected.

    Raises
    ------
    ValueError
        If arrays are not 1-D, have different lengths, or ``blank`` is
        missing when ``bg_correct="blank"``.

    Examples
    --------
    >>> import numpy as np
    >>> from GrowthCurve import n_at_t, summarize_growth
    >>> t = np.linspace(0, 24, 50)
    >>> n = n_at_t(k=0.5, n0=1e-5, r=1.2, t=t)
    >>> gc = summarize_growth(t, n)
    >>> gc.summary()
    """
    data_t = np.asarray(data_t, dtype=float)
    data_n = np.asarray(data_n, dtype=float)

    if data_t.ndim != 1 or data_n.ndim != 1:
        raise ValueError("data_t and data_n must be 1-D arrays.")
    if len(data_t) != len(data_n):
        raise ValueError("data_t and data_n must have the same length.")

    if bg_correct == "blank":
        if blank is None:
            raise ValueError(
                "A 'blank' array must be provided when bg_correct='blank'."
            )
        blank = np.asarray(blank, dtype=float)
        if len(blank) != len(data_n):
            raise ValueError("'blank' must have the same length as data_n.")

    # --- Time trimming ---
    # Vignette: "Measurements taken after this time should not be included"
    if t_trim > 0:
        mask = data_t < t_trim
        data_n = data_n[mask]
        data_t = data_t[mask]
        if bg_correct == "blank" and blank is not None:
            blank = blank[mask]

    t_max = float(np.max(data_t))

    # --- Background correction ---
    if bg_correct == "blank" and blank is not None:
        data_n = data_n - blank
        data_n = np.clip(data_n, 0.0, None)
    elif bg_correct == "min":
        data_n = data_n - np.min(data_n)
    # "none": do nothing

    # --- Fit logistic model ---
    fit_result = _fit_logistic(data_t, data_n)

    if fit_result is None:
        vals = GrowthVals(note="cannot fit data")
        return GrowthFit(vals=vals, model=None, data={"t": data_t, "N": data_n})

    popt, pcov = fit_result["popt"], fit_result["pcov"]
    df = fit_result["df"]
    k, n0, r = popt

    # --- Standard errors ---
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(3)
    k_se, n0_se, r_se = perr

    # --- Residual standard error (sigma) ---
    fitted = _logistic(data_t, k, n0, r)
    residuals = data_n - fitted
    sigma = float(np.sqrt(np.sum(residuals ** 2) / df)) if df > 0 else float("nan")

    # --- p-values ---
    k_p  = _compute_p_value(k,  k_se,  df)
    n0_p = _compute_p_value(n0, n0_se, df)
    r_p  = _compute_p_value(r,  r_se,  df)

    # --- Derived metrics ---
    t_inflection = t_at_inflection(k, n0, r)
    t_gen        = max_doubling_time(r)
    auc_l        = area_under_curve(k, n0, r, 0.0, t_max)
    auc_e        = empirical_auc(data_t, data_n, t_max)

    # --- Quality notes (mirror R logic from vignette) ---
    # "Growthcurver returns a note when K < N0, or when t_mid is negative"
    note = ""
    if k < n0:
        note = "questionable fit (k < n0)"
    elif t_inflection < 0:
        note = "questionable fit"

    vals = GrowthVals(
        k=float(k),       k_se=float(k_se),   k_p=float(k_p),
        n0=float(n0),     n0_se=float(n0_se), n0_p=float(n0_p),
        r=float(r),       r_se=float(r_se),   r_p=float(r_p),
        sigma=float(sigma), df=int(df),
        t_mid=float(t_inflection),
        t_gen=float(t_gen),
        auc_l=float(auc_l),
        auc_e=float(auc_e),
        note=note,
    )
    return GrowthFit(vals=vals, model=fit_result, data={"t": data_t, "N": data_n})


def summarize_growth_by_plate(
    plate_df: pd.DataFrame,
    t_trim: float = 0,
    bg_correct: str = "min",
) -> pd.DataFrame:
    """Fit logistic growth to every well/sample column in a plate DataFrame.

    Mirrors R's ``SummarizeGrowthByPlate()``, including the plate loop pattern
    described in the vignette's **"Get growth curves for a plate"** section.

    Input format (required, per vignette)
    --------------------------------------
    * A ``pandas.DataFrame`` where one column is named ``"time"``
      (case-insensitive) and contains the measurement times.
    * Each other column contains absorbance readings from a single well.
    * An optional column named ``"blank"`` may be included for blank-based
      background correction.
    * No missing values or non-numeric data.

    Parameters
    ----------
    plate_df:   DataFrame in the format described above.
    t_trim:     Discard data after this time (0 = use all data).
    bg_correct: ``"min"`` (default), ``"blank"``, or ``"none"``.

    Returns
    -------
    A :class:`pandas.DataFrame` with one row per sample/well and columns:
    ``sample``, ``k``, ``n0``, ``r``, ``t_mid``, ``t_gen``,
    ``auc_l``, ``auc_e``, ``sigma``, ``note``.

    The ``note`` column contains quality warnings (see :func:`summarize_growth`).
    Use :func:`quality_control` to automatically flag problematic wells.

    Raises
    ------
    ValueError
        If ``plate_df`` is not a DataFrame, has no ``"time"`` column, or
        ``bg_correct="blank"`` is selected but there is no ``"blank"`` column.

    Examples
    --------
    >>> summary = summarize_growth_by_plate(my_plate_df)
    >>> print(summary[summary["note"] != ""])   # show problematic wells
    """
    if not isinstance(plate_df, pd.DataFrame):
        raise ValueError("'plate_df' must be a pandas DataFrame.")

    col_lower = {c.lower(): c for c in plate_df.columns}
    if "time" not in col_lower:
        raise ValueError(
            "The DataFrame must contain a column named 'time' (case-insensitive)."
        )

    time_col = col_lower["time"]
    data_t = plate_df[time_col].to_numpy(dtype=float)

    blank_vec = None
    skip_cols = {time_col}

    if bg_correct == "blank":
        if "blank" not in col_lower:
            raise ValueError(
                "A 'blank' column is required when bg_correct='blank'."
            )
        blank_col = col_lower["blank"]
        blank_vec = plate_df[blank_col].to_numpy(dtype=float)
        skip_cols.add(blank_col)

    sample_cols = [c for c in plate_df.columns if c not in skip_cols]

    records = []
    for col in sample_cols:
        data_n = plate_df[col].to_numpy(dtype=float)
        try:
            gc = summarize_growth(
                data_t, data_n,
                t_trim=t_trim,
                bg_correct=bg_correct,
                blank=blank_vec,
            )
            v = gc.vals
            records.append({
                "sample": col,
                "k":      round(v.k,     6),
                "n0":     round(v.n0,    6),
                "r":      round(v.r,     6),
                "t_mid":  round(v.t_mid, 6),
                "t_gen":  round(v.t_gen, 6),
                "auc_l":  round(v.auc_l, 5),
                "auc_e":  round(v.auc_e, 5),
                "sigma":  round(v.sigma, 6),
                "note":   v.note,
            })
        except Exception as exc:  # noqa: BLE001
            records.append({
                "sample": col,
                "k": 0.0, "n0": 0.0, "r": 0.0,
                "t_mid": 0.0, "t_gen": 0.0,
                "auc_l": 0.0, "auc_e": 0.0,
                "sigma": 0.0,
                "note": f"error: {exc}",
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Quality control helpers  (vignette: "Quality control and best practices")
# ---------------------------------------------------------------------------

def quality_control(
    plate_summary: pd.DataFrame,
    sigma_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Flag problematic wells in a plate summary DataFrame.

    Implements the three quality-control steps from the vignette:

    1. **Fit notes** — any row where ``note != ""`` is flagged.
    2. **Sigma outliers** — rows whose ``sigma`` exceeds ``sigma_threshold``
       (if provided) or exceeds mean + 2*std of all sigma values.
    3. **Negative t_mid** — rows where ``t_mid < 0`` are flagged.

    Per the vignette:
    "You should look for outliers that have unusually large sigma values.
    Each sigma value is the residual sum of squares from the fit of the
    logistic curve to the data, so larger values mean poorer fits."

    Parameters
    ----------
    plate_summary:   Output DataFrame from :func:`summarize_growth_by_plate`.
    sigma_threshold: Optional manual threshold for flagging high sigma values.
                     If None, uses mean + 2 * std of all sigma values.

    Returns
    -------
    A copy of ``plate_summary`` with an additional boolean column
    ``"flagged"`` and a ``"qc_reason"`` string column explaining why
    each well was flagged (empty string if the well looks good).

    Examples
    --------
    >>> qc = quality_control(summary_df)
    >>> print(qc[qc["flagged"]])
    """
    df = plate_summary.copy()

    if sigma_threshold is None:
        sigma_mean = df["sigma"].mean()
        sigma_std  = df["sigma"].std()
        sigma_threshold = sigma_mean + 2 * sigma_std

    reasons = []
    for _, row in df.iterrows():
        r_list = []
        if row["note"] != "":
            r_list.append(f"fit note: '{row['note']}'")
        if row["sigma"] > sigma_threshold:
            r_list.append(f"high sigma ({row['sigma']:.4g} > {sigma_threshold:.4g})")
        if row["t_mid"] < 0:
            r_list.append("negative t_mid")
        reasons.append("; ".join(r_list))

    df["qc_reason"] = reasons
    df["flagged"]   = df["qc_reason"] != ""
    return df


def plot_sigma_histogram(
    plate_summary: pd.DataFrame,
    title: str = "Histogram of sigma values",
    show: bool = True,
):
    """Plot a histogram of sigma values to identify outlier wells.

    Directly mirrors the vignette code:
    ``hist(gc_out$sigma, main="Histogram of sigma values", xlab="sigma")``

    A well with a much larger sigma than others has a poor logistic fit and
    should be inspected manually.

    Parameters
    ----------
    plate_summary: Output DataFrame from :func:`summarize_growth_by_plate`.
    title:         Plot title.
    show:          If True (default), calls plt.show().

    Raises
    ------
    ImportError if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(plate_summary["sigma"].dropna(), bins="auto",
            color="steelblue", edgecolor="white")
    ax.set_xlabel("sigma")
    ax.set_ylabel("Count")
    ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_plate(
    plate_df: pd.DataFrame,
    plate_summary: Optional[pd.DataFrame] = None,
    t_trim: float = 0,
    bg_correct: str = "min",
    ncols: int = 12,
    fig_width: float = 22,
    fig_height: float = 15,
    show: bool = True,
):
    """Plot all wells in a plate in a grid layout (mirrors growthcurver PDF output).

    Mirrors the R vignette pattern:
    ``par(mfcol = c(8,12))`` + per-well SummarizeGrowth + plot loop.

    Parameters
    ----------
    plate_df:       Input plate DataFrame (with a ``"time"`` column).
    plate_summary:  Optional output from :func:`summarize_growth_by_plate`.
                    If provided, the fitted logistic curve is overlaid.
    t_trim:         Time trim used for fitting (for axis limit).
    bg_correct:     Background correction method.
    ncols:          Number of columns in the subplot grid (default 12).
    fig_width:      Figure width in inches.
    fig_height:     Figure height in inches.
    show:           If True (default), calls plt.show().

    Raises
    ------
    ImportError if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    col_lower = {c.lower(): c for c in plate_df.columns}
    time_col   = col_lower["time"]
    sample_cols = [c for c in plate_df.columns if c.lower() != "time"]

    nrows = math.ceil(len(sample_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # Build a lookup from plate_summary if provided
    summary_lookup = {}
    if plate_summary is not None:
        for _, row in plate_summary.iterrows():
            summary_lookup[row["sample"]] = row

    data_t = plate_df[time_col].to_numpy(dtype=float)

    for idx, col in enumerate(sample_cols):
        row_i = idx // ncols
        col_i = idx  % ncols
        ax = axes[row_i][col_i]

        data_n = plate_df[col].to_numpy(dtype=float)
        # Apply same bg correction for display
        if bg_correct == "min":
            data_n = data_n - np.min(data_n)

        ax.scatter(data_t, data_n, s=2, color="steelblue")

        # Overlay fit if available
        if col in summary_lookup:
            s = summary_lookup[col]
            if s["r"] > 0 and s["note"] == "":
                t_fit = np.linspace(data_t.min(), data_t.max(), 200)
                n_fit = n_at_t(s["k"], s["n0"], s["r"], t_fit)
                ax.plot(t_fit, n_fit, color="firebrick", linewidth=0.8)

        ax.set_title(col, fontsize=6, pad=2)
        ax.tick_params(labelsize=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(len(sample_cols), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if show:
        plt.show()

    return fig, axes