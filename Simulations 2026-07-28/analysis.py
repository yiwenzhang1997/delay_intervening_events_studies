"""
analysis.py
===========
High-level analysis functions for causal learning experiments.

Provides:
  1. run_analysis()    : run one or more models on one or more datasets,
                         with optional learning curves, DataFrame output,
                         and learning curve plots
  2. compare_datasets() : compare two run_analysis results with overlaid
                         learning curve plots and a summary table

LEARNING CURVES
---------------
The n_blocks parameter divides trials into equal checkpoints. With
n_blocks=4 and 24 trials, the model is fit at trials 6, 12, 18, and 24.
n_blocks=1 (default) fits only the full dataset.

For MCMC models (BayesianNoisyOr, GPGB), each checkpoint runs a separate
NUTS sampler on the trial subset up to that point.
For TD, checkpoints are read off w_history (one full run).
For Gallistel (GPI, GRI), the measures are recomputed on each subset.

LEARNING CURVE PLOT
-------------------
One panel per cause type (rows) x model (columns). Within each panel:
  - x-axis: trial number at checkpoint
  - y-axis: model estimate (free scale per panel — no extra white space)
  - one line per cause of that type
  - individual dataset lines shown at alpha_individual opacity
  - average line shown at alpha_avg opacity
  - true cs reference line (dashed) for BayesianNoisyOr C causes only

MULTIPLE DATASETS
-----------------
n_datasets > 1 generates datasets using seeds: seed, seed+1, seed+2, ...
Results are aggregated by averaging posterior means across datasets.

PRIMARY INTERFACE
-----------------
  results = run_analysis(
      dataset_kwargs = {...},
      models         = ["BayesianNoisyOr", "TD", "GPI", "GRI"],
      n_datasets     = 1,
      n_blocks       = 1,
      draws          = 2000,
      tune           = 1000,
      chains         = 4,
      target_accept  = 0.9,
      random_seed    = 42,
      alpha_avg      = 1.0,
      alpha_individual = 0.5,
      show_plots     = True,
      verbose        = True
  )

  results["individual_dataset_results"]  : list of per-dataset dicts
  results["aggregated_results"]          : averaged values + DataFrame + plot

Requires: numpy, pandas, matplotlib
          data_generation_and_utils, model_Bayesian_noisy_or,
          model_td, model_event_based, model_Gallistel_informativeness
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_generation_and_utils import (
    make_dataset,
    get_cause_columns,
    summarise_dataset,
    HOURS_PER_DAY,
)
from model_Bayesian_noisy_or import run_noisyor_model
from model_td import run_td_model
from model_Gong_et_al_2025 import run_gpgb_model
from model_Gallistel_informativeness import run_gallistel_model


# =============================================================================
# SECTION 1: CONSTANTS
# =============================================================================

VALID_MODELS = ["BayesianNoisyOr", "TD", "GPGB", "GPI", "GRI"]

_PALETTE = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#558B2F",
            "#00838F", "#AD1457", "#4E342E", "#37474F", "#F9A825"]

# Cause type display order for learning curve plots
_TYPE_ORDER = [
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


def _causes_of_type(type_key, cause_cols):
    if type_key == "C_only":
        return [c for c in cause_cols if re.match(r"^C\d{2}$", c)]
    return [c for c in cause_cols
            if not re.match(r"^C\d{2}$", c) and c != "BG"
            and c.startswith(type_key)]


# =============================================================================
# SECTION 2: SINGLE-DATASET, SINGLE-CHECKPOINT RUNNER
# =============================================================================

def _run_one_checkpoint(dataset, models, draws, tune, chains,
                         target_accept, random_seed, max_lag,
                         td_alpha, td_beta, td_gamma,
                         show_plots, verbose):
    """
    Run each requested model on a single dataset (or subset).
    GPI and GRI are handled via run_gallistel_model.
    Returns dict: model_name -> result dict (or None if failed).
    """
    out = {}
    need_gallistel = "GPI" in models or "GRI" in models

    # When show_plots=False, suppress all matplotlib output including any
    # plots generated internally by PyMC/ArviZ during sampling.
    if not show_plots:
        plt.ioff()
        plt.close("all")

    if "BayesianNoisyOr" in models:
        try:
            out["BayesianNoisyOr"] = run_noisyor_model(
                dataset       = dataset,
                draws         = draws,
                tune          = tune,
                chains        = chains,
                target_accept = target_accept,
                random_seed   = random_seed,
                show_plots    = show_plots,
                verbose       = verbose
            )
        except Exception as e:
            print(f"  WARNING: BayesianNoisyOr failed — {e}")
            out["BayesianNoisyOr"] = None

    if "TD" in models:
        try:
            out["TD"] = run_td_model(
                dataset    = dataset,
                alpha      = td_alpha,
                beta       = td_beta,
                gamma      = td_gamma,
                show_plots = show_plots,
                verbose    = verbose
            )
        except Exception as e:
            print(f"  WARNING: TD failed — {e}")
            out["TD"] = None

    if "GPGB" in models:
        try:
            out["GPGB"] = run_gpgb_model(
                dataset       = dataset,
                draws         = draws,
                tune          = tune,
                chains        = chains,
                target_accept = target_accept,
                random_seed   = random_seed,
                max_lag       = max_lag,
                show_plots    = show_plots,
                verbose       = verbose
            )
        except Exception as e:
            print(f"  WARNING: GPGB failed — {e}")
            out["GPGB"] = None

    if need_gallistel:
        try:
            gal = run_gallistel_model(dataset=dataset, verbose=verbose)
            if "GPI" in models:
                out["GPI"] = gal
            if "GRI" in models:
                out["GRI"] = gal
        except Exception as e:
            print(f"  WARNING: Gallistel failed — {e}")
            if "GPI" in models:
                out["GPI"] = None
            if "GRI" in models:
                out["GRI"] = None

    if not show_plots:
        plt.close("all")
        plt.ion()

    return out


# =============================================================================
# SECTION 3: EXTRACT ESTIMATES FROM MODEL RESULTS
# =============================================================================

def _extract_all_estimates(model_name, model_result, cause_cols):
    """
    Extract estimates for all causes from a model result.

    For BayesianNoisyOr / GPGB : posterior mean of cs per cause
    For TD                      : final associative weight per cause
    For GPI                     : GPI value per cause
    For GRI                     : GRI value per cause

    Returns dict: cause_name -> float (NaN on failure).
    """
    if model_result is None:
        return {col: np.nan for col in cause_cols}

    try:
        if model_name == "BayesianNoisyOr":
            idata = model_result["idata"]
            out   = {}
            for col in cause_cols:
                if col == "BG":
                    out["BG"] = float(idata.posterior["BG"].values.mean())
                else:
                    out[col] = float(
                        idata.posterior[f"cs_{col}"].values.mean()
                    )
            return out

        elif model_name == "TD":
            w_final = model_result["w_final"]
            td_cols = model_result["cause_cols"]
            return {col: float(w_final[i]) for i, col in enumerate(td_cols)}

        elif model_name == "GPGB":
            idata   = model_result["idata"]
            gp_cols = model_result["cause_cols"]
            cs_vals = idata.posterior["cs"].values
            return {col: float(cs_vals[:, :, i].mean())
                    for i, col in enumerate(gp_cols)}

        elif model_name == "GPI":
            return {col: float(model_result["GPI"].get(col, np.nan))
                    for col in model_result["cause_cols"]}

        elif model_name == "GRI":
            return {col: float(model_result["GRI"].get(col, np.nan))
                    for col in model_result["cause_cols"]}

    except Exception:
        pass

    return {col: np.nan for col in cause_cols}


def _y_label(model_name):
    """Return appropriate y-axis label for each model."""
    labels = {
        "BayesianNoisyOr": "Causal strength (cs)",
        "TD"             : "Associative weight (w)",
        "GPGB"           : "Causal strength (cs)",
        "GPI"            : "GPI",
        "GRI"            : "GRI",
    }
    return labels.get(model_name, "Estimate")


# =============================================================================
# SECTION 4: LEARNING CURVE LOGIC
# =============================================================================

def _run_learning_curve(dataset, models, n_blocks,
                         draws, tune, chains, target_accept, random_seed,
                         max_lag, td_alpha, td_beta, td_gamma):
    """
    Run each model at each of n_blocks checkpoints.

    TD: run once on full dataset, sample w_history at checkpoints.
    All others: refit MCMC on trial subset at each checkpoint.

    Returns
    -------
    learning_curves : dict, model_name -> list of {cause: estimate} per checkpoint
    checkpoints     : list of int trial counts
    """
    df_day  = dataset["day"]
    n_days  = len(df_day)
    tpb     = n_days // n_blocks
    checkpoints = [tpb * k for k in range(1, n_blocks + 1)]

    cause_cols      = [c for c in get_cause_columns(dataset) if c != "BG"]
    learning_curves = {m: [] for m in models}

    # TD: run once, read off w_history at each checkpoint
    if "TD" in models:
        td_full = run_td_model(
            dataset    = dataset,
            alpha      = td_alpha,
            beta       = td_beta,
            gamma      = td_gamma,
            show_plots = False,
            verbose    = False
        )
        td_cols = td_full["cause_cols"]
        for cp in checkpoints:
            row_idx = min(cp * HOURS_PER_DAY, td_full["w_history"].shape[0] - 1)
            w_at_cp = td_full["w_history"][row_idx]
            learning_curves["TD"].append(
                {col: float(w_at_cp[i]) for i, col in enumerate(td_cols)}
            )

    # All other models: refit on trial subset at each checkpoint
    other_models = [m for m in models if m != "TD"]
    if other_models:
        for cp in checkpoints:
            subset_day  = df_day.iloc[:cp].copy().reset_index(drop=True)
            subset_day["day"] = np.arange(1, cp + 1)
            subset_hour = dataset["hour"][
                dataset["hour"]["day"] <= cp
            ].copy().reset_index(drop=True)
            subset_hour["day"] = np.repeat(np.arange(1, cp + 1), HOURS_PER_DAY)

            subset = {
                "day"           : subset_day,
                "hour"          : subset_hour,
                "cause_metadata": dataset["cause_metadata"],
                "C01_hour"      : dataset["C01_hour"],
                "E_hour"        : dataset["E_hour"]
            }

            cp_results = _run_one_checkpoint(
                dataset       = subset,
                models        = other_models,
                draws         = draws,
                tune          = tune,
                chains        = chains,
                target_accept = target_accept,
                random_seed   = random_seed,
                max_lag       = max_lag,
                td_alpha      = td_alpha,
                td_beta       = td_beta,
                td_gamma      = td_gamma,
                show_plots    = False,
                verbose       = False
            )

            for m in other_models:
                estimates = _extract_all_estimates(
                    m, cp_results.get(m), cause_cols
                )
                learning_curves[m].append(estimates)

    return learning_curves, checkpoints


# =============================================================================
# SECTION 5: AGGREGATION AND DATAFRAME
# =============================================================================

def _aggregate_results(individual_results, checkpoints, models):
    """
    Average estimates across datasets at each checkpoint.

    Returns
    -------
    dict with keys:
        "mean_cs_by_checkpoint" : model -> list of {cause: mean} per checkpoint
        "mean_cs_final"         : model -> {cause: mean} at last checkpoint
        "checkpoints"           : list of int
        "n_datasets"            : int
        "learning_curve_df"     : long-format DataFrame with columns:
                                  model, cause, trial, estimate, dataset
                                  (dataset=0 means the average)
    """
    n_datasets    = len(individual_results)
    n_checkpoints = len(checkpoints)

    collected = {
        m: [{} for _ in range(n_checkpoints)]
        for m in models
    }

    for ds_idx, ds_result in enumerate(individual_results):
        lc = ds_result["learning_curves"]
        for m in models:
            if m not in lc:
                continue
            for ci, cs_dict in enumerate(lc[m]):
                for cause, val in cs_dict.items():
                    if cause not in collected[m][ci]:
                        collected[m][ci][cause] = []
                    collected[m][ci][cause].append(val)

    mean_cs_by_checkpoint = {m: [] for m in models}
    for m in models:
        for ci in range(n_checkpoints):
            mean_dict = {
                cause: float(np.nanmean(vals))
                for cause, vals in collected[m][ci].items()
            }
            mean_cs_by_checkpoint[m].append(mean_dict)

    mean_cs_final = {
        m: mean_cs_by_checkpoint[m][-1] if mean_cs_by_checkpoint[m] else {}
        for m in models
    }

    # Build long-format DataFrame
    rows = []
    # Individual dataset rows
    for ds_idx, ds_result in enumerate(individual_results):
        lc = ds_result["learning_curves"]
        for m in models:
            if m not in lc:
                continue
            for ci, cs_dict in enumerate(lc[m]):
                for cause, val in cs_dict.items():
                    rows.append({
                        "model"   : m,
                        "cause"   : cause,
                        "trial"   : checkpoints[ci],
                        "estimate": val,
                        "dataset" : ds_idx + 1
                    })
    # Average rows (dataset=0)
    for m in models:
        for ci, mean_dict in enumerate(mean_cs_by_checkpoint[m]):
            for cause, val in mean_dict.items():
                rows.append({
                    "model"   : m,
                    "cause"   : cause,
                    "trial"   : checkpoints[ci],
                    "estimate": val,
                    "dataset" : 0
                })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["model", "cause", "trial", "estimate", "dataset"]
    )

    return {
        "mean_cs_by_checkpoint": mean_cs_by_checkpoint,
        "mean_cs_final"        : mean_cs_final,
        "checkpoints"          : checkpoints,
        "n_datasets"           : n_datasets,
        "learning_curve_df"    : df
    }


# =============================================================================
# SECTION 6: LEARNING CURVE PLOT
# =============================================================================

def plot_learning_curves(aggregated, dataset, models,
                          alpha_avg, alpha_individual, show=True):
    """
    Plot learning curves as a grid: rows = cause types, columns = models.

    Within each panel:
      - x-axis: trial number at each checkpoint
      - y-axis: model estimate (free scale, no extra whitespace)
      - one line per cause in that type group
      - individual dataset lines at alpha_individual opacity
      - average line at alpha_avg opacity
      - dashed horizontal reference line = true cs for BayesianNoisyOr
        C cause panels only

    Parameters
    ----------
    aggregated       : dict from _aggregate_results()
    dataset          : dataset bundle (for cause metadata and true cs values)
    models           : list of model names
    alpha_avg        : float, opacity of average line (0-1)
    alpha_individual : float, opacity of individual dataset lines (0-1)
    show             : bool, call plt.show() (default True)

    Returns
    -------
    matplotlib Figure
    """
    df          = aggregated["learning_curve_df"]
    checkpoints = aggregated["checkpoints"]
    metadata    = dataset["cause_metadata"]

    cause_cols = [c for c in get_cause_columns(dataset) if c != "BG"]

    # Identify which cause types are present
    present_types = []
    for type_key, title in _TYPE_ORDER:
        cols = _causes_of_type(type_key, cause_cols)
        if cols:
            present_types.append((type_key, title, cols))

    if not present_types:
        print("No cause types to plot.")
        return None

    n_rows = len(present_types)
    n_cols = len(models)

    fig, axes = plt.subplots(
        nrows   = n_rows,
        ncols   = n_cols,
        figsize = (4.5 * n_cols, 3.5 * n_rows),
        squeeze = False
    )

    def _true_cs(cause):
        row = metadata[metadata["name"] == cause]
        if row.empty:
            return np.nan
        v = row.iloc[0]["true_cs"]
        return float(v) if not pd.isna(v) else np.nan

    n_datasets = aggregated["n_datasets"]

    for row_idx, (type_key, title, cols) in enumerate(present_types):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]

            model_df = df[df["model"] == model_name]

            all_vals = []

            for ci, cause in enumerate(cols):
                color = _PALETTE[ci % len(_PALETTE)]
                cause_df = model_df[model_df["cause"] == cause]

                # Individual dataset lines
                if n_datasets > 1 and alpha_individual > 0:
                    for ds in range(1, n_datasets + 1):
                        ds_df = cause_df[cause_df["dataset"] == ds].sort_values("trial")
                        if not ds_df.empty:
                            ax.plot(
                                ds_df["trial"], ds_df["estimate"],
                                color     = color,
                                alpha     = alpha_individual,
                                linewidth = 0.8,
                                linestyle = "-"
                            )
                            all_vals.extend(ds_df["estimate"].dropna().tolist())

                # Average line
                avg_df = cause_df[cause_df["dataset"] == 0].sort_values("trial")
                if not avg_df.empty:
                    ax.plot(
                        avg_df["trial"], avg_df["estimate"],
                        color     = color,
                        alpha     = alpha_avg,
                        linewidth = 2.0,
                        linestyle = "-",
                        label     = cause
                    )
                    all_vals.extend(avg_df["estimate"].dropna().tolist())

                # True cs reference line — BayesianNoisyOr, C causes only
                if model_name == "BayesianNoisyOr" and type_key == "C_only":
                    tc = _true_cs(cause)
                    if not np.isnan(tc):
                        ax.axhline(
                            y         = tc,
                            color     = color,
                            linewidth = 1.2,
                            linestyle = "--",
                            alpha     = 0.7
                        )
                        all_vals.append(tc)

            # Free y-axis: set limits snugly around the data
            if all_vals:
                ymin = np.nanmin(all_vals)
                ymax = np.nanmax(all_vals)
                pad  = max((ymax - ymin) * 0.1, 0.02)
                ax.set_ylim(ymin - pad, ymax + pad)

            ax.set_xlim(checkpoints[0] - 0.5, checkpoints[-1] + 0.5)
            ax.set_xticks(checkpoints)
            ax.tick_params(axis="both", labelsize=7)

            if row_idx == 0:
                ax.set_title(model_name, fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(title + "\n" + _y_label(model_name), fontsize=7)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Trial", fontsize=8)

            if len(cols) <= 8:
                ax.legend(fontsize=6, loc="best", ncol=1)

    fig.suptitle("Learning Curves", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if show:
        plt.show()

    return fig


# =============================================================================
# SECTION 7: TOP-LEVEL RUNNER
# =============================================================================

def run_analysis(dataset_kwargs,
                 models,
                 n_datasets,
                 n_blocks,
                 draws,
                 tune,
                 chains,
                 target_accept,
                 random_seed,
                 alpha_avg        = 1.0,
                 alpha_individual = 0.5,
                 max_lag          = 23,
                 td_alpha         = 0.2,
                 td_beta          = 0.8,
                 td_gamma         = 1.0,
                 show_plots       = True,
                 verbose          = True):
    """
    Run one or more models on one or more datasets with optional learning curves.

    Parameters
    ----------
    dataset_kwargs   : dict of kwargs for make_dataset(); seed key sets base seed
    models           : list of model names to run.
                       Valid: "BayesianNoisyOr", "TD", "GPGB", "GPI", "GRI"
    n_datasets       : int, N of datasets to generate (>= 1)
    n_blocks         : int, N of learning curve checkpoints (>= 1);
                       must evenly divide n_days
    draws            : int, NUTS posterior samples per chain
    tune             : int, NUTS tuning steps per chain
    chains           : int, N of MCMC chains
    target_accept    : float, NUTS target acceptance rate
    random_seed      : int, base random seed
    alpha_avg        : float, opacity of average line in learning curve plot (default 1.0)
    alpha_individual : float, opacity of individual dataset lines (default 0.5)
    max_lag          : int, max lag in hours for GPGB (default 23)
    td_alpha         : float, TD learning rate (default 0.2)
    td_beta          : float, TD trace decay (default 0.8)
    td_gamma         : float, TD discount factor (default 1.0)
    show_plots       : bool, display plots (default True)
    verbose          : bool, print progress (default True)

    Returns
    -------
    dict with keys:
        "individual_dataset_results" : list of n_datasets dicts, each with:
            "dataset"         : dataset bundle
            "learning_curves" : dict, model -> list of estimate dicts per checkpoint
            "final_results"   : dict, model -> full model result dict
            "dataset_seed"    : int
        "aggregated_results" : dict with keys:
            "mean_cs_by_checkpoint" : model -> list of mean estimate dicts
            "mean_cs_final"         : model -> mean estimate dict (final checkpoint)
            "checkpoints"           : list of int
            "n_datasets"            : int
            "learning_curve_df"     : long-format DataFrame
            "learning_curve_plot"   : matplotlib Figure (or None if n_blocks=1
                                      and show_plots=False)

    Raises
    ------
    ValueError if any model name is invalid, n_blocks does not divide n_days,
    n_datasets < 1, or n_blocks < 1.
    """
    invalid = [m for m in models if m not in VALID_MODELS]
    if invalid:
        raise ValueError(f"Unknown model(s): {invalid}. Valid: {VALID_MODELS}")
    if n_datasets < 1:
        raise ValueError(f"n_datasets must be >= 1, got {n_datasets}.")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}.")

    n_days = dataset_kwargs["n_days"]
    if n_days % n_blocks != 0:
        raise ValueError(
            f"n_blocks={n_blocks} does not evenly divide n_days={n_days}."
        )

    base_seed  = dataset_kwargs.get("seed", 42)
    individual = []

    for ds_idx in range(n_datasets):
        ds_seed   = base_seed + ds_idx
        ds_kwargs = dict(dataset_kwargs)
        ds_kwargs["seed"] = ds_seed

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Dataset {ds_idx + 1} / {n_datasets}  (seed={ds_seed})")
            print(f"{'='*60}")

        dataset = make_dataset(**ds_kwargs)

        if verbose and ds_idx == 0:
            summarise_dataset(dataset)

        show = show_plots and (ds_idx == n_datasets - 1)

        learning_curves, checkpoints = _run_learning_curve(
            dataset       = dataset,
            models        = models,
            n_blocks      = n_blocks,
            draws         = draws,
            tune          = tune,
            chains        = chains,
            target_accept = target_accept,
            random_seed   = ds_seed,
            max_lag       = max_lag,
            td_alpha      = td_alpha,
            td_beta       = td_beta,
            td_gamma      = td_gamma
        )

        # For final results: if n_blocks=1 we run once; if n_blocks>1 we
        # reuse the last checkpoint run to avoid a duplicate MCMC run.
        if n_blocks == 1:
            final_results = _run_one_checkpoint(
                dataset       = dataset,
                models        = models,
                draws         = draws,
                tune          = tune,
                chains        = chains,
                target_accept = target_accept,
                random_seed   = ds_seed,
                max_lag       = max_lag,
                td_alpha      = td_alpha,
                td_beta       = td_beta,
                td_gamma      = td_gamma,
                show_plots    = show,
                verbose       = verbose and ds_idx == 0
            )
        else:
            final_results = {m: None for m in models}

        individual.append({
            "dataset"        : dataset,
            "learning_curves": learning_curves,
            "final_results"  : final_results,
            "dataset_seed"   : ds_seed
        })

    aggregated = _aggregate_results(
        individual_results = individual,
        checkpoints        = checkpoints,
        models             = models
    )

    if verbose:
        print("\n=== Aggregated Results (posterior means, final checkpoint) ===")
        for m in models:
            final = aggregated["mean_cs_final"].get(m, {})
            if final:
                print(f"\n  {m}:")
                for cause, val in sorted(final.items()):
                    print(f"    {cause}: {val:.3f}")

    # Learning curve plot
    lc_fig = plot_learning_curves(
        aggregated       = aggregated,
        dataset          = individual[0]["dataset"],
        models           = models,
        alpha_avg        = alpha_avg,
        alpha_individual = alpha_individual,
        show             = show_plots
    )
    aggregated["learning_curve_plot"] = lc_fig

    return {
        "individual_dataset_results": individual,
        "aggregated_results"        : aggregated
    }


# =============================================================================
# SECTION 8: DATASET COMPARISON
# =============================================================================

def compare_datasets(results_list,
                     labels,
                     alpha_avg        = 1.0,
                     alpha_individual = 0.0,
                     show             = True):
    """
    Compare two or more run_analysis results objects: a summary table and
    a learning curve plot overlaying all conditions.

    All results objects must have been run with the same models list and
    the same cause structure. A ValueError is raised if they do not match.

    Colors are drawn from matplotlib's tab10 colormap — one distinct color
    per condition. When there are multiple causes of the same type, they
    are distinguished by linestyle (solid, dashed, dotted, dashdot).

    Individual dataset lines (from n_datasets > 1 runs) are shown at
    alpha_individual opacity. Default is 0.0 (hidden) since overlaying
    individual lines from multiple conditions gets very cluttered.

    The true cs reference line (black dashed) is shown for BayesianNoisyOr
    C cause panels only, taken from the first results object.

    Parameters
    ----------
    results_list     : list of dicts, each the output of run_analysis().
                       Must all use the same models and cause structure.
    labels           : list of str, one label per results object.
                       Must be the same length as results_list.
    alpha_avg        : float, opacity of average lines (default 1.0)
    alpha_individual : float, opacity of individual dataset lines (default 0.0)
    show             : bool, call plt.show() (default True)

    Returns
    -------
    table : pd.DataFrame, rows = models, columns = labels.
            Values are final-checkpoint mean estimates for C01.
    fig   : matplotlib Figure of overlaid learning curves.

    Raises
    ------
    ValueError if results_list and labels have different lengths, or if
    any results object has different models or cause columns from the first.
    """
    if len(results_list) != len(labels):
        raise ValueError(
            f"results_list has {len(results_list)} items but labels has "
            f"{len(labels)}. They must be the same length."
        )
    if len(results_list) < 2:
        raise ValueError("results_list must contain at least 2 results objects.")

    # Validate all results share the same models and causes
    ref_models = list(results_list[0]["aggregated_results"]["mean_cs_final"].keys())
    ref_causes = sorted(
        results_list[0]["aggregated_results"]["learning_curve_df"]["cause"]
        .unique().tolist()
    )
    for i, res in enumerate(results_list[1:], start=1):
        m = list(res["aggregated_results"]["mean_cs_final"].keys())
        if m != ref_models:
            raise ValueError(
                f"results_list[{i}] ('{labels[i]}') has models {m}, "
                f"but results_list[0] ('{labels[0]}') has {ref_models}."
            )

    models = ref_models

    # ------------------------------------------------------------------
    # Summary tables — one per cause, rows = models, columns = conditions
    # ------------------------------------------------------------------
    dataset0   = results_list[0]["individual_dataset_results"][0]["dataset"]
    # Use union of all cause columns across all conditions
    all_cause_cols = []
    seen = set()
    for res in results_list:
        ds      = res["individual_dataset_results"][0]["dataset"]
        ds_cols = [c for c in get_cause_columns(ds) if c != "BG"]
        for c in ds_cols:
            if c not in seen:
                all_cause_cols.append(c)
                seen.add(c)
    cause_cols = all_cause_cols

    tables = {}
    for cause in cause_cols:
        rows_table = {}
        for m in models:
            row = {}
            for res, label in zip(results_list, labels):
                agg = res["aggregated_results"]
                v   = agg["mean_cs_final"].get(m, {}).get(cause, np.nan)
                row[label] = round(float(v), 3)
            rows_table[m] = row
        t = pd.DataFrame(rows_table).T
        t.index.name = None
        tables[cause] = t

    print(f"\n{'='*60}")
    print(f"  Dataset Comparison — final estimates by cause")
    print(f"{'='*60}")
    for cause, t in tables.items():
        print(f"\n  {cause}:")
        print(t.to_string())

    # ------------------------------------------------------------------
    # Learning curve comparison plot
    # ------------------------------------------------------------------
    present_types = []
    for type_key, title in _TYPE_ORDER:
        cols = _causes_of_type(type_key, cause_cols)
        if cols:
            present_types.append((type_key, title, cols))

    if not present_types:
        print("No cause types to plot.")
        return tables, None

    n_plot_rows = len(present_types)
    n_plot_cols = len(models)

    fig, axes = plt.subplots(
        nrows   = n_plot_rows,
        ncols   = n_plot_cols,
        figsize = (4.5 * n_plot_cols, 3.5 * n_plot_rows),
        squeeze = False
    )

    metadata = dataset0["cause_metadata"]

    def _true_cs(cause):
        row = metadata[metadata["name"] == cause]
        if row.empty:
            return np.nan
        v = row.iloc[0]["true_cs"]
        return float(v) if not pd.isna(v) else np.nan

    # One color per condition from tab10
    import matplotlib.cm as cm
    tab10       = cm.get_cmap("tab10")
    cond_colors = [tab10(i % 10) for i in range(len(results_list))]

    # When a type has multiple causes, distinguish them by color shade
    # (darker/lighter within the condition color). For most cases there
    # will be only one cause per type per condition.
    # Linestyle is always solid — conditions are distinguished by color only.

    all_checkpoints = sorted(set(
        cp
        for res in results_list
        for cp in res["aggregated_results"]["checkpoints"]
    ))

    for row_idx, (type_key, title, cols) in enumerate(present_types):
        for col_idx, model_name in enumerate(models):
            ax       = axes[row_idx, col_idx]
            all_vals = []

            for ci, cause in enumerate(cols):
                # All lines are solid — color distinguishes conditions
                linestyle = "-"

                for res, label, color in zip(results_list, labels, cond_colors):
                    agg         = res["aggregated_results"]
                    df_lc       = agg["learning_curve_df"]
                    checkpoints = agg["checkpoints"]
                    n_ds        = agg["n_datasets"]

                    cause_df = df_lc[
                        (df_lc["model"] == model_name) &
                        (df_lc["cause"] == cause)
                    ]

                    # Individual lines
                    if n_ds > 1 and alpha_individual > 0:
                        for ds in range(1, n_ds + 1):
                            ds_df = cause_df[
                                cause_df["dataset"] == ds
                            ].sort_values("trial")
                            if not ds_df.empty:
                                ax.plot(
                                    ds_df["trial"], ds_df["estimate"],
                                    color     = color,
                                    alpha     = alpha_individual,
                                    linewidth = 0.7,
                                    linestyle = linestyle
                                )
                                all_vals.extend(
                                    ds_df["estimate"].dropna().tolist()
                                )

                    # Average line — only add legend entry if this cause
                    # exists in this condition's data
                    avg_df = cause_df[
                        cause_df["dataset"] == 0
                    ].sort_values("trial")
                    if not avg_df.empty:
                        lbl = label if len(cols) == 1 else f"{cause} — {label}"
                        ax.plot(
                            avg_df["trial"], avg_df["estimate"],
                            color     = color,
                            alpha     = alpha_avg,
                            linewidth = 2.0,
                            linestyle = linestyle,
                            label     = lbl
                        )
                        all_vals.extend(avg_df["estimate"].dropna().tolist())

                # True cs reference line — BayesianNoisyOr C causes only
                if model_name == "BayesianNoisyOr" and type_key == "C_only":
                    tc = _true_cs(cause)
                    if not np.isnan(tc):
                        ax.axhline(
                            y         = tc,
                            color     = "black",
                            linewidth = 1.0,
                            linestyle = "--",
                            alpha     = 0.4,
                            label     = f"true cs = {tc}"
                        )
                        all_vals.append(tc)

            # Free y-axis
            if all_vals:
                ymin = np.nanmin(all_vals)
                ymax = np.nanmax(all_vals)
                pad  = max((ymax - ymin) * 0.1, 0.02)
                ax.set_ylim(ymin - pad, ymax + pad)

            ax.set_xlim(all_checkpoints[0] - 0.5, all_checkpoints[-1] + 0.5)
            ax.set_xticks(all_checkpoints)
            ax.tick_params(axis="both", labelsize=7)

            if row_idx == 0:
                ax.set_title(model_name, fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(title + "\n" + _y_label(model_name), fontsize=7)
            if row_idx == n_plot_rows - 1:
                ax.set_xlabel("Trial", fontsize=8)

            ax.legend(fontsize=6, loc="best", ncol=1)

    condition_str = "  vs  ".join(labels)
    fig.suptitle(f"Dataset Comparison: {condition_str}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if show:
        plt.show()

    return tables, fig