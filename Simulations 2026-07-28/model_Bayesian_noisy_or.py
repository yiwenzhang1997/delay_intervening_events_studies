"""
model_Bayesian_noisy_or.py
==========================
Bayesian Noisy-OR model of causal strength.

Estimates the causal strength of each cause in a dataset using a noisy-OR
likelihood (Cheng, 1997) with delay-weighted Beta priors. Inference is via
MCMC (NUTS) using PyMC.

MODEL STRUCTURE
---------------
Three or more potential causes of E combine via noisy-OR:

  P(E=1) = 1 - prod_i[(1 - cs_i)^presence_i]

BG (background) is always present (column of 1s), so its term
(1 - cs_BG) always contributes, learning the baseline rate of E
independently of all observed causes.

PRIORS
------
  cs_i ~ Beta(1, 1 + K * delta_i)   delay-weighted; one per observed cause
  BG   ~ Uniform(0, 1)               background rate; no delay weighting

K is imported from data_generation_and_utils. The delay-weighted prior
means longer delays produce smaller prior means, encoding the assumption
that causes far from the effect in time are less likely to be causal.
Use plot_strength_prior() in data_generation_and_utils.py to visualise the
prior before fitting.

LIMITATION
----------
This model requires all causes to have fixed, known delays. Causes with
random delays (schedule == 'R') will raise a ValueError. Sporadic causes
(SIF, SBF, SAF) are accepted because they fire at one specific hour even
though that hour is chosen randomly at dataset generation time.

PRIMARY INTERFACE
-----------------
  results = run_noisyor_model(dataset, draws=2000, tune=1000, chains=4,
                              target_accept=0.9, random_seed=42,
                              show_plots=True, verbose=True)

  results["idata"]      : ArviZ InferenceData
  results["model"]      : PyMC model object
  results["summary"]    : posterior summary DataFrame
  results["diag"]       : MCMC diagnostics DataFrame
  results["plot"]       : matplotlib Figure
  results["cause_cols"] : list of cause column names in display order

Requires: pymc, arviz, pytensor, numpy, pandas, matplotlib, scipy
          data_generation_and_utils (same directory)
"""

import numpy as np
import pandas as pd
import re
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from data_generation_and_utils import (
    K,
    HOURS_PER_DAY,
    get_cause_columns,
    check_no_random_causes,
    get_elapsed,
)


# =============================================================================
# SECTION 1: MODEL — build and sample
# =============================================================================

def build_and_sample_noisyor(dataset, draws, tune, chains,
                              target_accept, random_seed):
    """
    Build and sample the BayesianNoisyOr model using NUTS.

    All parameters are required (no defaults) to encourage explicit calls.

    Noisy-OR likelihood (vectorized to keep the PyTensor graph flat):
      P(E=1) = 1 - prod_i[(1 - cs_i)^presence_i]

    BG is always 1, so its term (1 - cs_BG)^1 always contributes.
    The vectorized log-sum form avoids the C++ bracket-nesting compiler
    error that arises with many causes in a Python loop.

    Parameters
    ----------
    dataset       : dict, output of make_dataset()
    draws         : int, posterior samples per chain
    tune          : int, tuning (warm-up) steps per chain
    chains        : int, number of independent MCMC chains
    target_accept : float, NUTS target acceptance rate (0 < x < 1)
    random_seed   : int, for reproducibility

    Returns
    -------
    idata : ArviZ InferenceData
    model : PyMC model object
    """
    check_no_random_causes(dataset)

    df_day     = dataset["day"]
    cause_cols = get_cause_columns(dataset)
    E_obs      = df_day["E"].values.astype(float)
    elapsed    = get_elapsed(dataset)

    print("\n  Cause delays and priors:")
    for col in cause_cols:
        if col == "BG":
            print(f"    BG : always present  ->  Uniform(0, 1)")
            continue
        d     = elapsed[col]
        b     = 1.0 + K * d
        pmean = 1.0 / (1.0 + b)
        print(f"    {col}: delta = {d:.1f} hrs  ->  Beta(1, {b:.2f})"
              f"  prior mean = {pmean:.3f}")

    with pm.Model() as model:

        cs = {}
        for col in cause_cols:
            if col == "BG":
                cs["BG"] = pm.Uniform("BG", lower=0.0, upper=1.0)
            else:
                beta_b  = 1.0 + K * elapsed[col]
                cs[col] = pm.Beta(f"cs_{col}", alpha=1.0, beta=beta_b)

        # Vectorized noisy-OR in log space — avoids Python loop over causes
        # which would create deeply nested PyTensor graphs for many causes.
        presence_matrix = pt.as_tensor_variable(
            np.column_stack([df_day[col].values.astype(float)
                             for col in cause_cols])
        )  # shape: (n_trials, n_causes)

        log_one_minus_cs = pt.stack(
            [pt.log(1.0 - cs[col]) for col in cause_cols]
        )  # shape: (n_causes,)

        log_product = pt.dot(presence_matrix, log_one_minus_cs)
        p_effect    = pt.clip(1.0 - pt.exp(log_product), 1e-6, 1.0 - 1e-6)

        _ = pm.Bernoulli("E_obs", p=p_effect, observed=E_obs)

        idata = pm.sample(
            draws         = draws,
            tune          = tune,
            chains        = chains,
            target_accept = target_accept,
            random_seed   = random_seed,
            progressbar   = True
        )

    return idata, model


# =============================================================================
# SECTION 2: DIAGNOSTICS AND SUMMARY
# =============================================================================

def summarise_noisyor(idata, cause_cols):
    """
    Print and return the posterior summary for all causal strength parameters.

    Variable names in the summary:
      cs_C01, cs_CIF01_14, ... : Beta posteriors for observed causes
      BG                       : Uniform posterior for background

    Parameters
    ----------
    idata      : ArviZ InferenceData
    cause_cols : list of cause column names (from get_cause_columns)

    Returns
    -------
    pd.DataFrame : ArviZ summary table
    """
    vars_of_interest = [
        "BG" if c == "BG" else f"cs_{c}" for c in cause_cols
    ]
    summary = az.summary(idata, var_names=vars_of_interest, round_to=3)
    print("\n=== BayesianNoisyOr Posterior Summary ===")
    print(summary.to_string())
    return summary


def check_noisyor_diagnostics(idata, cause_cols):
    """
    Print and return MCMC convergence diagnostics.

    Targets: r_hat < 1.01, ess_bulk > 400. Prints a warning if either
    threshold is exceeded.

    Parameters
    ----------
    idata      : ArviZ InferenceData
    cause_cols : list of cause column names

    Returns
    -------
    pd.DataFrame : ArviZ diagnostics table
    """
    vars_of_interest = [
        "BG" if c == "BG" else f"cs_{c}" for c in cause_cols
    ]
    diag = az.summary(idata, var_names=vars_of_interest,
                      kind="diagnostics", round_to=3)
    print("\n=== BayesianNoisyOr MCMC Diagnostics ===")
    print(diag.to_string())

    r_hat_bad  = (diag["r_hat"] > 1.01).any()
    ess_bad    = (diag["ess_bulk"] < 400).any()

    if r_hat_bad:
        print("\nWARNING: Some r_hat values exceed 1.01. "
              "Consider increasing 'tune' or 'draws'.")
    if ess_bad:
        print("\nWARNING: Some ess_bulk values are below 400. "
              "Consider increasing 'draws'.")
    if not r_hat_bad and not ess_bad:
        print("\nAll diagnostics look good (r_hat < 1.01, ess_bulk > 400).")

    return diag


# =============================================================================
# SECTION 3: POSTERIOR PLOTS
# =============================================================================

# Bin edges for histograms — 0.0, 0.05, 0.10, ..., 1.0
BINS = list(np.arange(0, 1.05, 0.05))

# Cause type display order and labels (used in all plot functions)
_TYPE_ORDER = [
    ("C_only",  "C — main cause"),
    ("LCSF",    "LCSF — lure confound specified fixed"),
    ("LUSF",    "LUSF — lure uncorrelated specified fixed"),
    ("LCIF",    "LCIF — lure confound interim fixed"),
    ("LCIR",    "LCIR — lure confound interim random"),
    ("LCBF",    "LCBF — lure confound before fixed"),
    ("LCBR",    "LCBR — lure confound before random"),
    ("LCAR",    "LCAR — lure confound any random"),
    ("LUIF",    "LUIF — lure uncorrelated interim fixed"),
    ("LUIR",    "LUIR — lure uncorrelated interim random"),
    ("LUBF",    "LUBF — lure uncorrelated before fixed"),
    ("LUBR",    "LUBR — lure uncorrelated before random"),
    ("LUAR",    "LUAR — lure uncorrelated any random"),
    ("LSIF",    "LSIF — lure sporadic interim"),
    ("LSBF",    "LSBF — lure sporadic before"),
    ("LSAF",    "LSAF — lure sporadic any"),
    ("BG_only", "BG — background"),
]

_PALETTE = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#558B2F",
            "#00838F", "#AD1457", "#4E342E", "#37474F", "#F9A825"]


def _causes_of_type(type_key, cause_cols):
    """Return cause column names belonging to a given type key."""
    if type_key == "C_only":
        return [c for c in cause_cols if re.match(r"^C\d{2}$", c)]
    if type_key == "BG_only":
        return [c for c in cause_cols if c == "BG"]
    return [c for c in cause_cols
            if not re.match(r"^C\d{2}$", c) and c != "BG"
            and c.startswith(type_key)]


def _extract_samples(idata, var):
    """Flatten posterior samples for a named variable into a 1-D array."""
    return idata.posterior[var].values.flatten()


def plot_noisyor_posteriors(idata, cause_cols, dataset, show=True):
    """
    Plot posterior distributions of causal strength, grouped by cause type.

    Layout:
      - C causes: one panel, histogram (single cause) or overlaid KDE lines
        (multiple C causes). Always first.
      - Each other cause type (CS, CIF, CIR, ...): one panel with overlaid
        KDE lines, one line per cause. Empty types are omitted.
      - BG: one panel, histogram. Always last.
      - Panels arranged 4 columns wide, as many rows as needed.

    Parameters
    ----------
    idata      : ArviZ InferenceData
    cause_cols : list of cause column names in display order
    dataset    : dataset bundle (used to read cause_metadata)
    show       : bool, if True call plt.show() (default True)

    Returns
    -------
    matplotlib Figure
    """
    panels = []
    for type_key, title in _TYPE_ORDER:
        cols = _causes_of_type(type_key, cause_cols)
        if cols:
            panels.append((type_key, title, cols))

    n_panels = len(panels)
    N_COLS   = 4
    n_rows   = (n_panels + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(nrows=n_rows, ncols=N_COLS,
                             figsize=(5 * N_COLS, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, N_COLS)

    x_grid = np.linspace(0, 1, 300)

    for idx, (type_key, title, cols) in enumerate(panels):
        row = idx // N_COLS
        col = idx % N_COLS
        ax  = axes[row, col]

        if type_key == "BG_only":
            ax.hist(_extract_samples(idata, "BG"), bins=BINS,
                    color="#2196F3", alpha=0.6, edgecolor="#1565C0")
            ax.set_ylabel("Count")

        elif type_key == "C_only":
            if len(cols) == 1:
                ax.hist(_extract_samples(idata, f"cs_{cols[0]}"), bins=BINS,
                        color="#2196F3", alpha=0.6, edgecolor="#1565C0")
                ax.set_ylabel("Count")
            else:
                for j, c in enumerate(cols):
                    samp = _extract_samples(idata, f"cs_{c}")
                    kde  = gaussian_kde(samp)
                    ax.plot(x_grid, kde(x_grid),
                            color=_PALETTE[j % len(_PALETTE)],
                            linewidth=1.5, label=c)
                ax.set_ylabel("Density")
                ax.legend(fontsize=7)

        else:
            for j, c in enumerate(cols):
                samp = _extract_samples(idata, f"cs_{c}")
                kde  = gaussian_kde(samp)
                ax.plot(x_grid, kde(x_grid),
                        color=_PALETTE[j % len(_PALETTE)],
                        linewidth=1.5, label=c)
            ax.set_ylabel("Density")
            if len(cols) <= 10:
                ax.legend(fontsize=7, ncol=1)

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Causal strength")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # Hide unused axes
    for empty_idx in range(n_panels, n_rows * N_COLS):
        axes[empty_idx // N_COLS, empty_idx % N_COLS].set_visible(False)

    fig.suptitle("BayesianNoisyOr — Posterior Distributions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()
    return fig


# =============================================================================
# SECTION 4: TOP-LEVEL RUNNER
# =============================================================================

def run_noisyor_model(dataset, draws, tune, chains,
                      target_accept, random_seed,
                      show_plots=True, verbose=True):
    """
    Full BayesianNoisyOr pipeline: fit model, diagnostics, summary, plots.

    All MCMC parameters are required (no defaults) to ensure explicit calls.
    show_plots and verbose retain defaults since they are display options
    rather than analysis parameters.

    Parameters
    ----------
    dataset       : dict, output of make_dataset()
    draws         : int, posterior samples per chain
    tune          : int, tuning steps per chain
    chains        : int, number of MCMC chains
    target_accept : float, NUTS target acceptance rate
    random_seed   : int, for reproducibility
    show_plots    : bool, display posterior plots (default True)
    verbose       : bool, print dataset preview and priors (default True)

    Returns
    -------
    dict with keys:
        "idata"      : ArviZ InferenceData
        "model"      : PyMC model object
        "summary"    : pd.DataFrame, posterior summary (mean, sd, HDI, etc.)
        "diag"       : pd.DataFrame, MCMC diagnostics (r_hat, ess_bulk, etc.)
        "plot"       : matplotlib Figure of posterior distributions
        "cause_cols" : list of str, cause column names in display order
    """
    cause_cols = get_cause_columns(dataset)
    df_day     = dataset["day"]

    if verbose:
        print("=== BayesianNoisyOr Model ===")
        preview_cols = ["day"] + cause_cols + ["E"]
        print(df_day[preview_cols].head(10).to_string())
        print(f"\nTotal trials : {len(df_day)}")
        print(f"E=1 rate     : {df_day['E'].mean():.3f}")
        print(f"Causes       : {cause_cols}")
        print(f"K            : {K}")
        print("Fitting model with NUTS...")

    idata, model = build_and_sample_noisyor(
        dataset       = dataset,
        draws         = draws,
        tune          = tune,
        chains        = chains,
        target_accept = target_accept,
        random_seed   = random_seed
    )

    diag    = check_noisyor_diagnostics(idata, cause_cols) if verbose else None
    summary = summarise_noisyor(idata, cause_cols)         if verbose else None

    if show_plots:
        fig = plot_noisyor_posteriors(idata, cause_cols, dataset, show=True)
    else:
        fig = None

    return {
        "idata"     : idata,
        "model"     : model,
        "summary"   : summary,
        "diag"      : diag,
        "plot"      : fig,
        "cause_cols": cause_cols
    }
