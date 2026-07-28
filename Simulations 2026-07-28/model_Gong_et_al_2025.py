"""
model_Gong_et_al_2025.py
====================
Event-based and GPGB rate-based causal learning models.

Two models are implemented here:

  1. GPGB RATE-BASED MODEL (Gong, Pacer, Griffiths & Bramley 2025)
     Each cause contributes to a Poisson rate at each clock-hour via a
     normalized gamma delay density. Parameters: causal strength (cs) and
     delay distribution (shape, rate) per cause, plus background rate (BG).
     Inference via NUTS (PyMC). Handles both fixed and random-delay causes.

  2. EVENT-BASED MODEL (Gong et al. 2025, continuous-time version)
     Treats all days as a single continuous timeline. Cause tokens and
     effect events are matched via causal pathways. Two inference modes:
       - importance_sampling : marginal likelihoods for 2^N structures
       - nuts                : parameter estimation for the full structure

PRIMARY INTERFACES
------------------
  gpgb_results = run_gpgb_model(dataset, draws, tune, chains,
                                target_accept, random_seed,
                                max_lag=23, show_plots=True, verbose=True)

  eb_results = run_event_based_model(dataset,
                                     inference='importance_sampling', ...)

Requires: pymc, arviz, pytensor, numpy, pandas, scipy, matplotlib, re
          data_generation_and_utils (same directory)
"""

import numpy as np
import pandas as pd
import re
from itertools import product as iterproduct

import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.stats import gamma as scipy_gamma
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from data_generation_and_utils import (
    HOURS_PER_DAY,
    get_cause_columns,
)


# =============================================================================
# SECTION 1: GPGB — precomputation
# =============================================================================

def _build_gpgb_rate_matrix(dataset, max_lag):
    """
    Precompute the rate matrix needed by the GPGB model outside of PyMC.

    For each day d and clock-hour h (1..24), computes:
      - Observed effect counts: 1 at E_hour on days E occurred, 0 elsewhere.
      - For each cause i: the lag from cause i to hour h on day d, or NaN
        if cause i was absent that day or the lag is outside (0, max_lag].

    All parameters are required (no defaults).

    Parameters
    ----------
    dataset : dict, output of make_dataset()
    max_lag : int, maximum lag in hours to consider.
              Contributions with lag <= 0 or lag > max_lag are set to NaN.

    Returns
    -------
    dict with keys:
        "E_counts"       : np.ndarray, shape (n_days, 24), int.
        "lag_matrix"     : np.ndarray, shape (n_causes, n_days, 24), float.
                           NaN where cause absent or lag out of range.
        "cause_cols"     : list of cause column names (excludes BG)
        "cause_cols_all" : list including BG
        "n_days"         : int
        "E_hour"         : int
        "max_lag"        : int
    """
    df_day         = dataset["day"]
    E_hour         = dataset["E_hour"]
    cause_cols_all = get_cause_columns(dataset)
    cause_cols     = [c for c in cause_cols_all if c != "BG"]
    n_causes       = len(cause_cols)
    n_days         = len(df_day)
    hours          = np.arange(1, 25)

    # E_counts: shape (n_days, 24)
    E_counts  = np.zeros((n_days, 24), dtype=int)
    E_present = df_day["E"].values == 1
    for d in np.where(E_present)[0]:
        E_counts[d, E_hour - 1] = 1

    # Build cause_hour_by_day from the hour-level dataframe
    df_hour           = dataset["hour"]
    cause_hour_by_day = np.full((n_causes, n_days), np.nan)
    for ci, col in enumerate(cause_cols):
        for d in range(n_days):
            day_rows = df_hour[df_hour["day"] == d + 1]
            fired    = day_rows[day_rows[col] == 1]["hour"].values
            if len(fired) > 0:
                cause_hour_by_day[ci, d] = fired[0]

    # lag_matrix[i, d, h] = hours[h] - cause_hour_by_day[i, d]
    cause_hour_3d = cause_hour_by_day[:, :, np.newaxis]   # (n_causes, n_days, 1)
    hours_3d      = hours[np.newaxis, np.newaxis, :]       # (1, 1, 24)
    lag_matrix    = hours_3d - cause_hour_3d               # (n_causes, n_days, 24)

    invalid = (np.isnan(lag_matrix) |
               (lag_matrix <= 0)    |
               (lag_matrix > max_lag))
    lag_matrix[invalid] = np.nan

    return {
        "E_counts"       : E_counts,
        "lag_matrix"     : lag_matrix,
        "cause_cols"     : cause_cols,
        "cause_cols_all" : cause_cols_all,
        "n_days"         : n_days,
        "E_hour"         : E_hour,
        "max_lag"        : max_lag
    }


# =============================================================================
# SECTION 2: GPGB — model, diagnostics, summary, plots
# =============================================================================

def build_and_sample_gpgb(dataset, draws, tune, chains,
                           target_accept, random_seed, max_lag):
    """
    Build and sample the GPGB rate-based model via NUTS.

    All parameters are required (no defaults).

    MODEL STRUCTURE
    ---------------
    Each cause i contributes to the Poisson rate at (day d, hour h) via:

      mydec(lag; shape, rate) = (lag/mode)^(shape-1) * exp((shape-1) - rate*lag)

    where mode = (shape-1)/rate (clamped to 1e-6). This normalizes so that
    cs_i is interpretable as the peak rate contribution at the mode delay.

    Total rate:
      lambda(d,h) = BG + sum_i [cs_i * mydec(lag_i(d,h); shape_i, rate_i)]

    Observed counts ~ Poisson(lambda(d,h)).

    PRIORS
    ------
    cs_i         ~ Beta(1, 1)        causal strength per cause
    shape_delay  ~ Gamma(3, 1)       shape of delay distribution
    rate_delay   ~ Gamma(1, 1.5)     rate of delay distribution
    BG           ~ Exponential(1)    background rate per hour

    DERIVED
    -------
    mode_delay = clip((shape-1)/rate, 0, inf)   most likely delay (hours)

    Parameters
    ----------
    dataset       : dict, output of make_dataset()
    draws         : int, posterior samples per chain
    tune          : int, tuning steps per chain
    chains        : int, number of MCMC chains
    target_accept : float, NUTS target acceptance rate
    random_seed   : int, for reproducibility
    max_lag       : int, maximum lag in hours

    Returns
    -------
    idata   : ArviZ InferenceData
    model   : PyMC model object
    precomp : dict from _build_gpgb_rate_matrix()
    """
    precomp    = _build_gpgb_rate_matrix(dataset, max_lag=max_lag)
    E_counts   = precomp["E_counts"]
    lag_matrix = precomp["lag_matrix"]
    cause_cols = precomp["cause_cols"]
    n_causes   = len(cause_cols)

    valid_mask = (~np.isnan(lag_matrix)).astype(float)
    lag_safe   = np.where(np.isnan(lag_matrix), 1.0, lag_matrix)

    print(f"\n  GPGB model: {n_causes} causes + BG")
    print(f"  Observed bins: {E_counts.shape[0]} days x 24 hours")
    print(f"  Effect observations: {E_counts.sum()}")

    with pm.Model() as model:

        cs    = pm.Beta("cs",          alpha=1.0, beta=1.0, shape=n_causes)
        shape = pm.Gamma("shape_delay", alpha=3.0, beta=1.0, shape=n_causes)
        rate  = pm.Gamma("rate_delay",  alpha=1.0, beta=1.5, shape=n_causes)
        BG    = pm.Exponential("BG",    lam=1.0)

        mode_delay = pm.Deterministic(
            "mode_delay",
            pt.clip((shape - 1.0) / rate, 0.0, np.inf)
        )

        # Reshape for broadcasting (n_causes, n_days, 24)
        cs_3d    = cs[:, np.newaxis, np.newaxis]
        shape_3d = shape[:, np.newaxis, np.newaxis]
        rate_3d  = rate[:, np.newaxis, np.newaxis]
        mode_3d  = pt.clip((shape_3d - 1.0) / rate_3d, 1e-6, np.inf)

        lag_t   = pt.as_tensor_variable(lag_safe)
        valid_t = pt.as_tensor_variable(valid_mask)

        # log(mydec) = (shape-1)*(log(lag) - log(mode) + 1) - rate*lag
        log_mydec = ((shape_3d - 1.0) * (pt.log(lag_t) - pt.log(mode_3d) + 1.0)
                     - rate_3d * lag_t)
        mydec     = pt.exp(log_mydec) * valid_t

        lambda_matrix = pt.clip(
            pt.sum(cs_3d * mydec, axis=0) + BG, 1e-9, np.inf
        )

        _ = pm.Poisson("E_obs", mu=lambda_matrix, observed=E_counts)

        idata = pm.sample(
            draws         = draws,
            tune          = tune,
            chains        = chains,
            target_accept = target_accept,
            random_seed   = random_seed,
            progressbar   = True
        )

    return idata, model, precomp


def summarise_gpgb(idata, cause_cols):
    """
    Print and return posterior summary for GPGB model parameters.

    Parameters
    ----------
    idata      : ArviZ InferenceData
    cause_cols : list of cause column names (excludes BG)

    Returns
    -------
    pd.DataFrame
    """
    vars_of_interest = ["cs", "shape_delay", "rate_delay", "mode_delay", "BG"]
    summary = az.summary(idata, var_names=vars_of_interest, round_to=3)

    rename = {}
    for j, col in enumerate(cause_cols):
        rename[f"cs[{j}]"]          = f"cs_{col}"
        rename[f"shape_delay[{j}]"] = f"shape_delay_{col}"
        rename[f"rate_delay[{j}]"]  = f"rate_delay_{col}"
        rename[f"mode_delay[{j}]"]  = f"mode_delay_{col}"
    summary = summary.rename(index=rename)

    print("\n=== GPGB Posterior Summary ===")
    print(summary.to_string())
    return summary


def check_gpgb_diagnostics(idata):
    """
    Print and return MCMC convergence diagnostics for the GPGB model.

    Targets: r_hat < 1.01, ess_bulk > 400.

    Parameters
    ----------
    idata : ArviZ InferenceData

    Returns
    -------
    pd.DataFrame
    """
    vars_of_interest = ["cs", "shape_delay", "rate_delay", "BG"]
    diag = az.summary(idata, var_names=vars_of_interest,
                      kind="diagnostics", round_to=3)
    print("\n=== GPGB MCMC Diagnostics ===")
    print(diag.to_string())

    r_hat_bad = (diag["r_hat"] > 1.01).any()
    ess_bad   = (diag["ess_bulk"] < 400).any()

    if r_hat_bad:
        print("\nWARNING: Some r_hat values exceed 1.01. "
              "Consider increasing 'tune' or 'draws'.")
    if ess_bad:
        print("\nWARNING: Some ess_bulk values are below 400. "
              "Consider increasing 'draws'.")
    if not r_hat_bad and not ess_bad:
        print("\nAll diagnostics look good (r_hat < 1.01, ess_bulk > 400).")
    return diag


_TYPE_ORDER_GPGB = [
    ("C_only", "C — main cause"),
    ("LCSF",   "LCSF — lure confound specified fixed"),
    ("LUSF",   "LUSF — lure uncorrelated specified fixed"),
    ("LCIF",   "LCIF — lure confound interim fixed"),
    ("LCIR",   "LCIR — lure confound interim random"),
    ("LCBF",   "LCBF — lure confound before fixed"),
    ("LCBR",   "LCBR — lure confound before random"),
    ("LCAR",   "LCAR — lure confound any random"),
    ("LUIF",   "LUIF — lure uncorrelated interim fixed"),
    ("LUIR",   "LUIR — lure uncorrelated interim random"),
    ("LUBF",   "LUBF — lure uncorrelated before fixed"),
    ("LUBR",   "LUBR — lure uncorrelated before random"),
    ("LUAR",   "LUAR — lure uncorrelated any random"),
    ("LSIF",   "LSIF — lure sporadic interim"),
    ("LSBF",   "LSBF — lure sporadic before"),
    ("LSAF",   "LSAF — lure sporadic any"),
]

_PALETTE = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#558B2F",
            "#00838F", "#AD1457", "#4E342E", "#37474F", "#F9A825"]


def _causes_of_type_gpgb(type_key, cause_cols):
    if type_key == "C_only":
        return [c for c in cause_cols if re.match(r"^C\d{2}$", c)]
    return [c for c in cause_cols
            if not re.match(r"^C\d{2}$", c) and c.startswith(type_key)]


def plot_gpgb_posteriors(idata, cause_cols, dataset, show=True):
    """
    Plot GPGB posterior distributions grouped by cause type.

    For each non-empty cause type, two panels are shown side by side:
      Left:  cs (causal strength) KDE lines, one per cause
      Right: mode_delay KDE lines, one per cause
    BG gets a single KDE panel at the end.
    Layout: 4 columns wide, as many rows as needed.

    Parameters
    ----------
    idata      : ArviZ InferenceData from build_and_sample_gpgb()
    cause_cols : list of cause column names (excludes BG)
    dataset    : dict, dataset bundle
    show       : bool, if True call plt.show() (default True)

    Returns
    -------
    matplotlib Figure
    """
    type_panels = []
    for type_key, title in _TYPE_ORDER_GPGB:
        cols = _causes_of_type_gpgb(type_key, cause_cols)
        if cols:
            type_panels.append((type_key, title, cols))

    n_types  = len(type_panels)
    n_panels = 2 * n_types + 1   # cs + mode per type, plus BG
    N_COLS   = 4
    n_rows   = (n_panels + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(nrows=n_rows, ncols=N_COLS,
                             figsize=(5 * N_COLS, 4 * n_rows))
    axes = np.array(axes).reshape(n_rows, N_COLS)

    x_cs   = np.linspace(0, 1, 300)
    x_mode = np.linspace(0, 24, 300)

    cs_samples         = idata.posterior["cs"].values         # (chains, draws, n)
    mode_delay_samples = idata.posterior["mode_delay"].values
    BG_samples         = idata.posterior["BG"].values.flatten()

    def _flat(arr, j):
        return arr[:, :, j].flatten()

    panel_idx = 0
    for type_key, title, cols in type_panels:

        # cs panel
        row = panel_idx // N_COLS
        col = panel_idx % N_COLS
        ax  = axes[row, col]
        panel_idx += 1
        for j_col, c in enumerate(cols):
            c_idx   = cause_cols.index(c)
            samples = _flat(cs_samples, c_idx)
            kde     = gaussian_kde(samples)
            ax.plot(x_cs, kde(x_cs), color=_PALETTE[j_col % len(_PALETTE)],
                    linewidth=1.5, label=c)
        ax.set_ylabel("Density")
        if len(cols) <= 10:
            ax.legend(fontsize=7)
        ax.set_title(f"{title}\nCausal strength (cs)", fontsize=8,
                     fontweight="bold")
        ax.set_xlabel("cs")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

        # mode_delay panel
        row = panel_idx // N_COLS
        col = panel_idx % N_COLS
        ax  = axes[row, col]
        panel_idx += 1
        for j_col, c in enumerate(cols):
            c_idx   = cause_cols.index(c)
            samples = _flat(mode_delay_samples, c_idx)
            kde     = gaussian_kde(samples)
            ax.plot(x_mode, kde(x_mode), color=_PALETTE[j_col % len(_PALETTE)],
                    linewidth=1.5, label=c)
        ax.set_ylabel("Density")
        if len(cols) <= 10:
            ax.legend(fontsize=7)
        ax.set_title(f"{title}\nMode delay (hrs)", fontsize=8,
                     fontweight="bold")
        ax.set_xlabel("mode_delay (hours)")
        ax.set_xlim(0, 24)

    # BG panel
    row = panel_idx // N_COLS
    col = panel_idx % N_COLS
    ax  = axes[row, col]
    bg_kde = gaussian_kde(BG_samples)
    x_bg   = np.linspace(0, np.percentile(BG_samples, 99) * 1.1, 300)
    ax.plot(x_bg, bg_kde(x_bg), color="#2196F3", linewidth=1.5)
    ax.set_title("BG — background rate\n(events per hour)", fontsize=8,
                 fontweight="bold")
    ax.set_xlabel("BG rate")
    ax.set_ylabel("Density")
    panel_idx += 1

    for empty in range(panel_idx, n_rows * N_COLS):
        axes[empty // N_COLS, empty % N_COLS].set_visible(False)

    fig.suptitle("GPGB Rate-Based Model — Posterior Distributions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()
    return fig


def run_gpgb_model(dataset, draws, tune, chains,
                   target_accept, random_seed,
                   max_lag=23, show_plots=True, verbose=True):
    """
    Full GPGB pipeline: precompute rate matrix, fit model, diagnostics,
    summary, plots.

    All MCMC parameters are required (no defaults). max_lag, show_plots,
    and verbose retain defaults since they are display/tuning options.

    Parameters
    ----------
    dataset       : dict, output of make_dataset()
    draws         : int, posterior samples per chain
    tune          : int, tuning steps per chain
    chains        : int, number of MCMC chains
    target_accept : float, NUTS target acceptance rate
    random_seed   : int, for reproducibility
    max_lag       : int, maximum lag in hours (default 23)
    show_plots    : bool, display posterior plots (default True)
    verbose       : bool, print dataset preview (default True)

    Returns
    -------
    dict with keys:
        "idata"      : ArviZ InferenceData
        "model"      : PyMC model object
        "summary"    : pd.DataFrame, posterior summary
        "diag"       : pd.DataFrame, MCMC diagnostics
        "plot"       : matplotlib Figure
        "cause_cols" : list of str, cause column names (excludes BG)
        "precomp"    : dict, precomputed rate matrix arrays
    """
    cause_cols = [c for c in get_cause_columns(dataset) if c != "BG"]
    df_day     = dataset["day"]

    if verbose:
        print("=== GPGB Rate-Based Model ===")
        print(f"Total trials : {len(df_day)}")
        print(f"E=1 rate     : {df_day['E'].mean():.3f}")
        print(f"Causes       : {cause_cols}")
        print(f"max_lag      : {max_lag} hours")
        print("Fitting model with NUTS...")

    idata, model, precomp = build_and_sample_gpgb(
        dataset       = dataset,
        draws         = draws,
        tune          = tune,
        chains        = chains,
        target_accept = target_accept,
        random_seed   = random_seed,
        max_lag       = max_lag
    )

    diag    = check_gpgb_diagnostics(idata)          if verbose else None
    summary = summarise_gpgb(idata, cause_cols)       if verbose else None

    if show_plots:
        fig = plot_gpgb_posteriors(idata, cause_cols, dataset, show=True)
    else:
        fig = None

    return {
        "idata"     : idata,
        "model"     : model,
        "summary"   : summary,
        "diag"      : diag,
        "plot"      : fig,
        "cause_cols": cause_cols,
        "precomp"   : precomp
    }


# =============================================================================
# SECTION 3: EVENT-BASED MODEL — shared infrastructure
# =============================================================================

def _to_absolute_hours(dataset):
    """
    Convert the day-level dataset to a single continuous timeline.

    Day 1 occupies hours 0-23, day 2 hours 24-47, etc.

    Returns
    -------
    events     : list of dicts with keys 'type', 'name', 'abs_hour', 'day'
    cause_cols : list of observed cause column names (excludes BG)
    t_end      : float, absolute hour of last observed event
    """
    df_day     = dataset["day"]
    cause_cols = [c for c in get_cause_columns(dataset) if c != "BG"]
    E_hour     = dataset["E_hour"]
    n_days     = len(df_day)

    events = []
    for _, row in df_day.iterrows():
        day        = int(row["day"])
        day_offset = (day - 1) * HOURS_PER_DAY

        for col in cause_cols:
            if row[col] == 1:
                elapsed_col = f"{col}_elapsed"
                if elapsed_col in df_day.columns:
                    elapsed = row[elapsed_col]
                    if pd.notna(elapsed):
                        cause_hour = E_hour - elapsed
                        events.append({
                            "type"    : "cause",
                            "name"    : col,
                            "abs_hour": day_offset + cause_hour,
                            "day"     : day
                        })

        if row["E"] == 1:
            events.append({
                "type"    : "effect",
                "name"    : "E",
                "abs_hour": day_offset + E_hour,
                "day"     : day
            })

    events.sort(key=lambda x: x["abs_hour"])
    t_end = (max(e["abs_hour"] for e in events)
             if events else float(n_days * HOURS_PER_DAY))
    return events, cause_cols, t_end


def _enumerate_pathways_for_causes(events, active_causes, t_end, max_delay):
    """
    Enumerate valid causal pathways for a given set of active causes.

    Each individual cause firing is a distinct token. For each effect event,
    candidates are: background ('B') plus any token that fired within
    max_delay before the effect. One-cause-one-effect constraint enforced:
    no token appears in more than one effect assignment.

    All parameters are required (no defaults).

    Returns
    -------
    effect_times   : list of float
    token_times    : dict, token name -> float
    cause_of_token : dict, token name -> cause column name
    pathways       : list of tuples (one element per effect)
    bg_prev_times  : list of float (previous effect time per effect; 0 for first)
    """
    effect_events = [e for e in events if e["type"] == "effect"]
    effect_times  = [e["abs_hour"] for e in effect_events]
    n_effects     = len(effect_times)

    token_times    = {}
    cause_of_token = {}
    for col in active_causes:
        firings = sorted([e["abs_hour"] for e in events
                          if e["type"] == "cause" and e["name"] == col])
        for idx, t_c in enumerate(firings):
            tok = f"{col}_t{idx:02d}"
            token_times[tok]    = t_c
            cause_of_token[tok] = col

    if n_effects == 0:
        return [], token_times, cause_of_token, [], []

    candidates = []
    for t_e in effect_times:
        cands = ["B"]
        for tok, t_c in token_times.items():
            delay = t_e - t_c
            if 0 < delay <= max_delay:
                cands.append(tok)
        candidates.append(cands)

    bg_prev_times = [0.0] + effect_times[:-1]

    pathways = []
    for pw in iterproduct(*candidates):
        non_bg = [x for x in pw if x != "B"]
        if len(non_bg) == len(set(non_bg)):
            pathways.append(pw)

    return effect_times, token_times, cause_of_token, pathways, bg_prev_times


def _event_based_loglik(w_c_vals, mu_vals, sigma2_vals, br_val,
                        effect_times, token_times, cause_of_token,
                        pathways, bg_prev_times, active_causes, t_end):
    """
    Log-likelihood of observed events under the event-based model for a
    single parameter draw, following Gong et al. (2025).

    Returns
    -------
    float : log-likelihood (-1e30 if no valid pathways or zero total prob)
    """
    if not pathways:
        return -1e30

    alpha = {}
    beta  = {}
    for col in active_causes:
        mu     = max(float(mu_vals[col]),     1e-10)
        sigma2 = max(float(sigma2_vals[col]), 1e-10)
        alpha[col] = (mu ** 2) / sigma2
        beta[col]  = mu / sigma2

    pathway_probs = []
    for pw in pathways:
        prob = 1.0

        for i, assignee in enumerate(pw):
            if assignee == "B":
                delay = effect_times[i] - bg_prev_times[i]
                p = float(br_val) * np.exp(-float(br_val) * delay)
            else:
                col   = cause_of_token[assignee]
                t_c   = token_times[assignee]
                delay = effect_times[i] - t_c
                if delay <= 0:
                    prob = 0.0
                    break
                p = (scipy_gamma.pdf(delay,
                                     a=alpha[col],
                                     scale=1.0 / beta[col])
                     * float(w_c_vals[col]))
            if p <= 0.0:
                prob = 0.0
                break
            prob *= p

        if prob > 0.0:
            assigned_tokens = set(x for x in pw if x != "B")
            for tok, t_c in token_times.items():
                if tok in assigned_tokens:
                    continue
                col       = cause_of_token[tok]
                remaining = t_end - t_c
                if remaining <= 0:
                    miss = 1.0 - float(w_c_vals[col])
                else:
                    surv = 1.0 - scipy_gamma.cdf(
                        remaining, a=alpha[col], scale=1.0 / beta[col]
                    )
                    miss = (1.0 - float(w_c_vals[col])) + float(w_c_vals[col]) * surv
                prob *= miss

        pathway_probs.append(prob)

    total = sum(pathway_probs)
    return np.log(total) if total > 0.0 else -1e30


# =============================================================================
# SECTION 4: EVENT-BASED MODEL — importance sampling
# =============================================================================

def _make_structure_table(cause_cols):
    """Build binary indicator matrix A (2^N, N) and structure labels."""
    N         = len(cause_cols)
    n_structs = 2 ** N
    A         = np.zeros((n_structs, N), dtype=int)
    labels    = []
    for k in range(n_structs):
        bits   = [(k >> j) & 1 for j in range(N)]
        A[k]   = bits
        labels.append("S" + "".join(str(b) for b in bits))
    return A, labels


def _sample_prior_is(m, cause_cols, t_end, max_mean_delay, seed):
    """Draw M samples from the joint prior (all parameters required)."""
    rng    = np.random.default_rng(seed)
    N      = len(cause_cols)
    w_c    = rng.uniform(0.0, 1.0,            size=(m, N))
    mu     = rng.uniform(0.0, max_mean_delay, size=(m, N))
    u_sig  = rng.uniform(0.0, 1.0,            size=(m, N))
    sigma2 = u_sig * (mu ** 2)
    u_br   = rng.uniform(0.0, t_end,          size=(m,))
    br     = 1.0 / np.maximum(u_br, 1e-10)
    return {"w_c": w_c, "mu": mu, "sigma2": sigma2, "br": br}


def _compute_ess(log_weights):
    """Compute ESS from log-unnormalised weights."""
    log_w_norm = log_weights - logsumexp(log_weights)
    w_norm     = np.exp(log_w_norm)
    return 1.0 / np.sum(w_norm ** 2)


def _run_importance_sampling(events, cause_cols, t_end,
                              max_delay, max_mean_delay, m, seed):
    """
    Run importance sampling for model comparison across all 2^N structures.

    All parameters are required (no defaults).

    Returns
    -------
    dict with keys:
        'log_marglik', 'posterior_struct', 'posterior_cause',
        'param_estimates', 'ess', 'A', 'labels',
        'log_lik_matrix', 'prior_samples', 'cause_cols'
    """
    N         = len(cause_cols)
    n_structs = 2 ** N

    A, labels = _make_structure_table(cause_cols)
    prior     = _sample_prior_is(m, cause_cols, t_end,
                                  max_mean_delay=max_mean_delay, seed=seed)

    print(f"  Enumerating pathways for {n_structs} structures...")
    pathway_cache = {}
    for k in range(n_structs):
        active = [cause_cols[j] for j in range(N) if A[k, j] == 1]
        et, tt, cot, pw, bg = _enumerate_pathways_for_causes(
            events, active, t_end, max_delay=max_delay
        )
        pathway_cache[k] = {
            "effect_times"   : et,
            "token_times"    : tt,
            "cause_of_token" : cot,
            "pathways"       : pw,
            "bg_prev_times"  : bg,
            "active_causes"  : active
        }
        print(f"    {labels[k]:10s}: {len(active)} causes, {len(pw)} pathways")

    print(f"\n  Computing likelihood matrix ({m} samples x {n_structs} structures)...")
    log_lik = np.full((m, n_structs), -1e30)

    for i in range(m):
        if i % 1000 == 0:
            print(f"    Sample {i}/{m}...")

        w_c_i    = {cause_cols[j]: prior["w_c"][i, j]    for j in range(N)}
        mu_i     = {cause_cols[j]: prior["mu"][i, j]     for j in range(N)}
        sigma2_i = {cause_cols[j]: prior["sigma2"][i, j] for j in range(N)}
        br_i     = prior["br"][i]

        for k in range(n_structs):
            pc     = pathway_cache[k]
            active = pc["active_causes"]
            w_c_k    = {c: w_c_i[c]    for c in active}
            mu_k     = {c: mu_i[c]     for c in active}
            sigma2_k = {c: sigma2_i[c] for c in active}

            log_lik[i, k] = _event_based_loglik(
                w_c_vals       = w_c_k,
                mu_vals        = mu_k,
                sigma2_vals    = sigma2_k,
                br_val         = br_i,
                effect_times   = pc["effect_times"],
                token_times    = pc["token_times"],
                cause_of_token = pc["cause_of_token"],
                pathways       = pc["pathways"],
                bg_prev_times  = pc["bg_prev_times"],
                active_causes  = active,
                t_end          = t_end
            )

    log_marglik = logsumexp(log_lik, axis=0) - np.log(m)
    log_post    = log_marglik - logsumexp(log_marglik)
    post_struct = np.exp(log_post)
    post_cause  = post_struct @ A
    ess         = np.array([_compute_ess(log_lik[:, k]) for k in range(n_structs)])

    k_full     = n_structs - 1
    log_w      = log_lik[:, k_full]
    log_w_norm = log_w - logsumexp(log_w)
    w_norm     = np.exp(log_w_norm)

    param_estimates = {}
    for j, col in enumerate(cause_cols):
        param_estimates[f"w_c_{col}"]    = float(np.dot(w_norm, prior["w_c"][:, j]))
        param_estimates[f"mu_{col}"]     = float(np.dot(w_norm, prior["mu"][:, j]))
        param_estimates[f"sigma2_{col}"] = float(np.dot(w_norm, prior["sigma2"][:, j]))
    param_estimates["br"] = float(np.dot(w_norm, prior["br"]))

    return {
        "log_marglik"     : log_marglik,
        "posterior_struct": post_struct,
        "posterior_cause" : post_cause,
        "param_estimates" : param_estimates,
        "ess"             : ess,
        "A"               : A,
        "labels"          : labels,
        "log_lik_matrix"  : log_lik,
        "prior_samples"   : prior,
        "cause_cols"      : cause_cols
    }


def _print_is_results(is_results):
    """Print a formatted summary of importance sampling results."""
    cause_cols  = is_results["cause_cols"]
    labels      = is_results["labels"]
    post_struct = is_results["posterior_struct"]
    post_cause  = is_results["posterior_cause"]
    ess         = is_results["ess"]
    param_est   = is_results["param_estimates"]
    N           = len(cause_cols)

    print("\n=== Event-Based Model: Structure Posteriors ===")
    print(f"{'Structure':<12}", end="")
    for col in cause_cols:
        print(f"{col:>8}", end="")
    print(f"{'P(S|d)':>10}  {'ESS':>8}")
    print("-" * (12 + 8 * N + 20))

    for k, (label, p, e) in enumerate(zip(labels, post_struct, ess)):
        A_row = is_results["A"][k]
        print(f"{label:<12}", end="")
        for bit in A_row:
            print(f"{'yes' if bit else 'no':>8}", end="")
        print(f"{p:>10.4f}  {e:>8.1f}")

    print("\n=== Per-Cause Marginal Posterior P(cause present) ===")
    for j, col in enumerate(cause_cols):
        print(f"  {col}: {post_cause[j]:.4f}")

    print("\n=== Weighted Parameter Estimates (full model) ===")
    if ess[-1] < 100:
        print(f"  WARNING: ESS = {ess[-1]:.1f} — estimates may be unreliable.")
        print(f"  Consider increasing m.")
    for k, v in param_est.items():
        print(f"  {k}: {v:.4f}")

    for k, (label, e) in enumerate(zip(labels, ess)):
        if e < 100:
            print(f"  WARNING: ESS = {e:.1f} for {label} — may be unreliable.")


# =============================================================================
# SECTION 5: EVENT-BASED MODEL — NUTS inference
# =============================================================================

def _run_nuts(dataset, cause_cols, events, t_end,
              max_delay, max_mean_delay, draws, tune, chains,
              target_accept, random_seed):
    """
    Fit the fully-connected event-based model via PyMC / NUTS.

    All parameters are active (no structure search). Good for parameter
    estimation; model comparison requires bridge sampling (not implemented).

    All parameters are required (no defaults).

    Returns
    -------
    idata : ArviZ InferenceData
    model : PyMC model object
    """
    from pytensor.graph.op import Op

    print("\n  NOTE: NUTS mode fits all causes as present.")
    print("  Good for parameter estimation.")
    print("  Model comparison requires bridge sampling (not implemented).\n")

    active = cause_cols
    et, tt, cot, pw, bg = _enumerate_pathways_for_causes(
        events, active, t_end, max_delay=max_delay
    )
    print(f"  Valid pathways: {len(pw)}")
    if not pw:
        raise ValueError(
            "No valid pathways. Check that causes precede effects "
            "and that max_delay is sufficient."
        )

    N = len(cause_cols)

    class EventBasedLogLikOp(Op):
        itypes = [pt.dscalar] * (3 * N + 1)
        otypes = [pt.dscalar]

        def perform(self, node, inputs, outputs):
            w_c_vals    = {cause_cols[j]: float(inputs[j])       for j in range(N)}
            mu_vals     = {cause_cols[j]: float(inputs[N + j])   for j in range(N)}
            sigma2_vals = {cause_cols[j]: float(inputs[2*N + j]) for j in range(N)}
            br_val      = float(inputs[3*N])

            ll = _event_based_loglik(
                w_c_vals       = w_c_vals,
                mu_vals        = mu_vals,
                sigma2_vals    = sigma2_vals,
                br_val         = br_val,
                effect_times   = et,
                token_times    = tt,
                cause_of_token = cot,
                pathways       = pw,
                bg_prev_times  = bg,
                active_causes  = active,
                t_end          = t_end
            )
            outputs[0][0] = np.float64(ll)

    loglik_op = EventBasedLogLikOp()

    with pm.Model() as model:
        w_c    = [pm.Uniform(f"w_c_{cause_cols[j]}", lower=0.0, upper=1.0)
                  for j in range(N)]
        mu     = [pm.Uniform(f"mu_{cause_cols[j]}",
                             lower=0.0, upper=max_mean_delay)
                  for j in range(N)]
        sigma2 = [pm.Uniform(f"sigma2_{cause_cols[j]}",
                             lower=0.0, upper=mu[j] ** 2)
                  for j in range(N)]
        u_bg   = pm.Uniform("u_bg", lower=0.0, upper=t_end)
        br     = pm.Deterministic("br", 1.0 / u_bg)

        for j in range(N):
            pm.Deterministic(
                f"alpha_{cause_cols[j]}",
                mu[j] ** 2 / pt.maximum(sigma2[j], 1e-10)
            )
            pm.Deterministic(
                f"beta_{cause_cols[j]}",
                mu[j] / pt.maximum(sigma2[j], 1e-10)
            )

        param_tensors = (
            [w_c[j].astype("float64")    for j in range(N)] +
            [mu[j].astype("float64")     for j in range(N)] +
            [sigma2[j].astype("float64") for j in range(N)] +
            [br.astype("float64")]
        )
        ll_val = loglik_op(*param_tensors)
        pm.Potential("event_based_loglik", ll_val)

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
# SECTION 6: EVENT-BASED MODEL — top-level runner
# =============================================================================

def run_event_based_model(dataset,
                           inference,
                           m              = 10000,
                           max_delay      = 48.0,
                           max_mean_delay = 100.0,
                           override_n     = False,
                           draws          = 2000,
                           tune           = 1000,
                           chains         = 4,
                           target_accept  = 0.9,
                           random_seed    = 42,
                           verbose        = True):
    """
    Run the event-based causal learning model (Gong et al. 2025).

    Treats all days as one continuous timeline. inference is required
    (no default) to prevent accidental use of one mode over the other.

    Parameters
    ----------
    dataset        : dict, output of make_dataset()
    inference      : str, 'importance_sampling' or 'nuts' (required)
    m              : int, prior samples for importance sampling (default 10000)
    max_delay      : float, pathway trim in hours (default 48.0)
    max_mean_delay : float, upper bound on mu prior (default 100.0)
    override_n     : bool, allow N > 8 causes for IS (default False)
    draws          : int, NUTS posterior samples per chain (default 2000)
    tune           : int, NUTS tuning steps (default 1000)
    chains         : int, NUTS chains (default 4)
    target_accept  : float, NUTS target acceptance (default 0.9)
    random_seed    : int (default 42)
    verbose        : bool (default True)

    Returns
    -------
    For inference='importance_sampling':
        dict with keys: 'is_results', 'cause_cols', 'events', 't_end'

    For inference='nuts':
        dict with keys: 'idata', 'model', 'cause_cols', 'events', 't_end'
    """
    events, cause_cols, t_end = _to_absolute_hours(dataset)
    N = len(cause_cols)

    if verbose:
        df_day    = dataset["day"]
        n_effects = sum(1 for e in events if e["type"] == "effect")
        print("=== Event-Based Model (Continuous Timeline) ===")
        print(f"  Inference     : {inference}")
        print(f"  Cause columns : {cause_cols}  (N={N})")
        print(f"  t_end         : {t_end:.1f} hours")
        print(f"  max_delay     : {max_delay:.1f} hours")
        print(f"  max_mean_delay: {max_mean_delay:.1f} hours")
        print(f"  Effect events : {n_effects}")
        print(f"  E=1 rate      : {df_day['E'].mean():.2f}")

    if inference == "importance_sampling":
        if N > 8 and not override_n:
            raise ValueError(
                f"N={N} observed causes would require 2^{N}={2**N} structures. "
                f"This may be very slow. Set override_n=True to proceed anyway."
            )

        is_results = _run_importance_sampling(
            events         = events,
            cause_cols     = cause_cols,
            t_end          = t_end,
            max_delay      = max_delay,
            max_mean_delay = max_mean_delay,
            m              = m,
            seed           = random_seed
        )

        if verbose:
            _print_is_results(is_results)

        return {
            "is_results": is_results,
            "cause_cols": cause_cols,
            "events"    : events,
            "t_end"     : t_end
        }

    elif inference == "nuts":
        idata, model = _run_nuts(
            dataset        = dataset,
            cause_cols     = cause_cols,
            events         = events,
            t_end          = t_end,
            max_delay      = max_delay,
            max_mean_delay = max_mean_delay,
            draws          = draws,
            tune           = tune,
            chains         = chains,
            target_accept  = target_accept,
            random_seed    = random_seed
        )

        if verbose:
            vars_of_interest = (
                [f"w_c_{col}"    for col in cause_cols] +
                [f"mu_{col}"     for col in cause_cols] +
                [f"sigma2_{col}" for col in cause_cols] +
                ["br"]
            )
            present = [v for v in vars_of_interest
                       if v in idata.posterior.data_vars]
            summary = az.summary(idata, var_names=present, round_to=3)
            print("\n=== Event-Based Model (NUTS) Posterior Summary ===")
            print(summary.to_string())

            problems = []
            for var in present:
                if var not in summary.index:
                    continue
                if summary.loc[var, "r_hat"] > 1.05:
                    problems.append(
                        f"  WARNING: {var} r_hat = {summary.loc[var, 'r_hat']:.3f}"
                    )
                if summary.loc[var, "ess_bulk"] < 200:
                    problems.append(
                        f"  WARNING: {var} ess_bulk = "
                        f"{summary.loc[var, 'ess_bulk']:.0f}"
                    )
            if problems:
                for p in problems:
                    print(p)
            else:
                print("  Diagnostics OK: r_hat <= 1.05, ess_bulk >= 200")

        return {
            "idata"     : idata,
            "model"     : model,
            "cause_cols": cause_cols,
            "events"    : events,
            "t_end"     : t_end
        }

    else:
        raise ValueError(
            f"inference must be 'importance_sampling' or 'nuts', "
            f"got '{inference}'"
        )


def run_structure_inference(dataset, m, max_delay, max_mean_delay,
                             override_n, random_seed):
    """
    Convenience wrapper: run importance sampling and display structure posteriors.

    All parameters are required (no defaults) to encourage explicit calls.

    Returns
    -------
    dict from run_event_based_model()
    """
    return run_event_based_model(
        dataset        = dataset,
        inference      = "importance_sampling",
        m              = m,
        max_delay      = max_delay,
        max_mean_delay = max_mean_delay,
        override_n     = override_n,
        random_seed    = random_seed,
        verbose        = True
    )