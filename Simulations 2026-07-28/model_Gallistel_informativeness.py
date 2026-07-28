"""
model_Gallistel_informativeness.py
===================================
Gallistel Prospective Informativeness (GPI) and Retrospective Informativeness
(GRI) models of causal learning.

Based on the informativeness ratio framework from Gallistel et al. (working
paper), equations 1a and 1b. These are non-Bayesian, non-parametric measures
that compute causal relevance from the temporal statistics of cause-effect
co-occurrence on a continuous hour-level timeline — ignoring trial/day
structure entirely.

MEASURES (per cause C, relative to effect E)
--------------------------------------------

GPI (Prospective Informativeness):
  The factor by which C's occurrence increases the expected rate of E.
  
  GPI = mean_inter_E_interval / mean_C_to_next_E_interval
      = M_EE_time / M_CE_time

  Where:
    M_CE_time : mean time from each C occurrence to the NEXT E occurrence
                (in absolute hours across all days). If C occurs but no E
                follows before the end of the dataset, that C is excluded.
    M_EE_time : mean time from each E occurrence to the NEXT E occurrence.
                If an E has no subsequent E, it is excluded.

GRI (Retrospective Informativeness):
  The factor by which E's occurrence increases the estimated rate of C
  looking backward.

  GRI = mean_inter_C_interval / mean_E_to_prior_C_interval
      = M_CC_time / M_EC_time

  Where:
    M_EC_time : mean time from each E back to the MOST RECENT C preceding it
                (in absolute hours). If an E has no prior C in the dataset,
                that E is excluded.
    M_CC_time : mean time from each C back to the most recent prior C.
                If a C has no prior C, it is excluded.

All intervals are computed in absolute hours using the absolute_hour column
in the hour-level dataframe — day boundaries are irrelevant.

GPI and GRI are computed independently for every cause in the dataset
(excluding BG). A value > 1 means the cause is informative about the effect
(or vice versa). A value = 1 means the cause provides no temporal information.

MISSING DATA HANDLING
---------------------
When an interval cannot be computed (e.g., C occurs but no subsequent E
exists before end of dataset), that observation is treated as missing and
excluded from the mean. If ALL observations are missing for a cause, GPI or
GRI for that cause is NaN.

PRIMARY INTERFACE
-----------------
  results = run_gallistel_model(dataset, verbose=True)

  results["GPI"]        : dict, cause_name -> float (or NaN)
  results["GRI"]        : dict, cause_name -> float (or NaN)
  results["details"]    : dict, cause_name -> dict of intermediate values
  results["cause_cols"] : list of cause column names (BG excluded)

References
----------
Gallistel, C. R., et al. (working paper). A simple, transparent and useable
  formalization of associative learning, rooted in information theory.
  Equations 1a and 1b.

Requires: numpy, pandas
          data_generation_and_utils (same directory)
"""

import numpy as np
import pandas as pd

from data_generation_and_utils import (
    HOURS_PER_DAY,
    get_cause_columns,
)


# =============================================================================
# SECTION 1: ABSOLUTE HOUR EXTRACTION
# =============================================================================

def _get_absolute_event_times(dataset, cause_col):
    """
    Extract absolute hour timestamps for a single cause and for E.

    Uses the absolute_hour column from the hour-level dataframe, which runs
    continuously across all days: day 1 hour 1 = 1, day 1 hour 24 = 24,
    day 2 hour 1 = 25, etc.

    Parameters
    ----------
    dataset   : dict, output of make_dataset()
    cause_col : str, name of the cause column (not BG)

    Returns
    -------
    cause_times : np.ndarray of float, sorted absolute hours of C occurrences
    effect_times: np.ndarray of float, sorted absolute hours of E occurrences
    """
    df_hour = dataset["hour"]

    cause_rows  = df_hour[df_hour[cause_col] == 1]
    effect_rows = df_hour[df_hour["E"] == 1]

    cause_times  = np.sort(cause_rows["absolute_hour"].values.astype(float))
    effect_times = np.sort(effect_rows["absolute_hour"].values.astype(float))

    return cause_times, effect_times


# =============================================================================
# SECTION 2: INTERVAL CALCULATIONS
# =============================================================================

def _compute_CE_intervals(cause_times, effect_times):
    """
    Compute time from each C occurrence to the NEXT E occurrence.

    Intervals are computed in absolute hours. If a C has no subsequent E
    in the dataset, that C is excluded (treated as missing).

    Parameters
    ----------
    cause_times  : np.ndarray, sorted absolute hours of C occurrences
    effect_times : np.ndarray, sorted absolute hours of E occurrences

    Returns
    -------
    np.ndarray of float, one interval per C that has a subsequent E.
    Empty array if no valid intervals exist.
    """
    intervals = []
    for t_c in cause_times:
        # Find the earliest E that occurs strictly after this C
        subsequent = effect_times[effect_times > t_c]
        if len(subsequent) > 0:
            intervals.append(subsequent[0] - t_c)
    return np.array(intervals)


def _compute_EE_intervals(effect_times):
    """
    Compute time from each E occurrence to the NEXT E occurrence.

    The last E has no subsequent E and is excluded.

    Parameters
    ----------
    effect_times : np.ndarray, sorted absolute hours of E occurrences

    Returns
    -------
    np.ndarray of float (length = n_effects - 1), or empty array.
    """
    if len(effect_times) < 2:
        return np.array([])
    return np.diff(effect_times)


def _compute_EC_intervals(cause_times, effect_times):
    """
    Compute time from each E occurrence back to the MOST RECENT PRIOR C.

    Intervals are computed in absolute hours (positive values: E - C).
    If an E has no prior C in the dataset, that E is excluded.

    Parameters
    ----------
    cause_times  : np.ndarray, sorted absolute hours of C occurrences
    effect_times : np.ndarray, sorted absolute hours of E occurrences

    Returns
    -------
    np.ndarray of float, one interval per E that has a prior C.
    Empty array if no valid intervals exist.
    """
    intervals = []
    for t_e in effect_times:
        # Find the most recent C that occurred strictly before this E
        prior = cause_times[cause_times < t_e]
        if len(prior) > 0:
            intervals.append(t_e - prior[-1])
    return np.array(intervals)


def _compute_CC_intervals(cause_times):
    """
    Compute time from each C occurrence back to the MOST RECENT PRIOR C.

    Equivalent to inter-C intervals (differences between consecutive C times).
    The first C has no prior C and is excluded.

    Parameters
    ----------
    cause_times : np.ndarray, sorted absolute hours of C occurrences

    Returns
    -------
    np.ndarray of float (length = n_causes - 1), or empty array.
    """
    if len(cause_times) < 2:
        return np.array([])
    return np.diff(cause_times)


# =============================================================================
# SECTION 3: GPI AND GRI COMPUTATION
# =============================================================================

def compute_gpi_gri(dataset, cause_col):
    """
    Compute GPI and GRI for a single cause column.

    All interval calculations use absolute hours from the hour-level
    dataframe. Day boundaries are ignored entirely.

    Parameters
    ----------
    dataset   : dict, output of make_dataset()
    cause_col : str, name of the cause column (not BG)

    Returns
    -------
    dict with keys:
        "GPI"           : float or NaN
        "GRI"           : float or NaN
        "M_CE_time"     : float, mean time from C to next E (hours)
        "M_EE_time"     : float, mean inter-E interval (hours)
        "M_EC_time"     : float, mean time from E back to most recent C (hours)
        "M_CC_time"     : float, mean inter-C interval (hours)
        "n_CE"          : int, number of valid C->E intervals
        "n_EE"          : int, number of valid E->E intervals
        "n_EC"          : int, number of valid E->C intervals
        "n_CC"          : int, number of valid C->C intervals
        "cause_col"     : str
    """
    cause_times, effect_times = _get_absolute_event_times(dataset, cause_col)

    CE_intervals = _compute_CE_intervals(cause_times, effect_times)
    EE_intervals = _compute_EE_intervals(effect_times)
    EC_intervals = _compute_EC_intervals(cause_times, effect_times)
    CC_intervals = _compute_CC_intervals(cause_times)

    M_CE = float(np.mean(CE_intervals)) if len(CE_intervals) > 0 else np.nan
    M_EE = float(np.mean(EE_intervals)) if len(EE_intervals) > 0 else np.nan
    M_EC = float(np.mean(EC_intervals)) if len(EC_intervals) > 0 else np.nan
    M_CC = float(np.mean(CC_intervals)) if len(CC_intervals) > 0 else np.nan

    # GPI = M_EE / M_CE  (ratio > 1 means C predicts E faster than baseline)
    if np.isnan(M_EE) or np.isnan(M_CE) or M_CE == 0:
        GPI = np.nan
    else:
        GPI = M_EE / M_CE

    # GRI = M_CC / M_EC  (ratio > 1 means E retrodicts C faster than baseline)
    if np.isnan(M_CC) or np.isnan(M_EC) or M_EC == 0:
        GRI = np.nan
    else:
        GRI = M_CC / M_EC

    return {
        "GPI"      : GPI,
        "GRI"      : GRI,
        "M_CE_time": M_CE,
        "M_EE_time": M_EE,
        "M_EC_time": M_EC,
        "M_CC_time": M_CC,
        "n_CE"     : len(CE_intervals),
        "n_EE"     : len(EE_intervals),
        "n_EC"     : len(EC_intervals),
        "n_CC"     : len(CC_intervals),
        "cause_col": cause_col
    }


# =============================================================================
# SECTION 4: TOP-LEVEL RUNNER
# =============================================================================

def run_gallistel_model(dataset, verbose=True):
    """
    Compute GPI and GRI for every cause in the dataset (excluding BG).

    Parameters
    ----------
    dataset : dict, output of make_dataset()
    verbose : bool, print results table (default True)

    Returns
    -------
    dict with keys:
        "GPI"        : dict, cause_name -> float (GPI value, or NaN)
        "GRI"        : dict, cause_name -> float (GRI value, or NaN)
        "details"    : dict, cause_name -> full detail dict from
                       compute_gpi_gri() for that cause
        "cause_cols" : list of str, cause column names (BG excluded)
    """
    cause_cols = [c for c in get_cause_columns(dataset) if c != "BG"]

    GPI     = {}
    GRI     = {}
    details = {}

    for col in cause_cols:
        result     = compute_gpi_gri(dataset=dataset, cause_col=col)
        GPI[col]   = result["GPI"]
        GRI[col]   = result["GRI"]
        details[col] = result

    if verbose:
        print("=== Gallistel Informativeness Model ===")
        print(f"\n  {'Cause':<18} {'GPI':>8} {'GRI':>8} "
              f"{'M_CE':>8} {'M_EE':>8} {'M_EC':>8} {'M_CC':>8} "
              f"{'n_CE':>6} {'n_EE':>6} {'n_EC':>6} {'n_CC':>6}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} "
              f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} "
              f"{'-'*6} {'-'*6} {'-'*6} {'-'*6}")

        for col in cause_cols:
            d = details[col]

            def _fmt(x):
                return f"{x:8.3f}" if not np.isnan(x) else "     NaN"

            print(
                f"  {col:<18}"
                f"{_fmt(d['GPI'])}"
                f"{_fmt(d['GRI'])}"
                f"{_fmt(d['M_CE_time'])}"
                f"{_fmt(d['M_EE_time'])}"
                f"{_fmt(d['M_EC_time'])}"
                f"{_fmt(d['M_CC_time'])}"
                f"{d['n_CE']:>6}"
                f"{d['n_EE']:>6}"
                f"{d['n_EC']:>6}"
                f"{d['n_CC']:>6}"
            )

        print("\n  GPI = M_EE / M_CE  (> 1: C predicts E faster than baseline)")
        print("  GRI = M_CC / M_EC  (> 1: E retrodicts C faster than baseline)")
        print("  NaN: insufficient data to compute the interval.")

    return {
        "GPI"       : GPI,
        "GRI"       : GRI,
        "details"   : details,
        "cause_cols": cause_cols
    }
