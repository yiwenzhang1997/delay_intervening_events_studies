"""
data_generation_and_utils.py
============================
Data generation and shared utilities for causal learning experiments.

This module is the entry point for all analyses. Call make_dataset() to
generate a dataset bundle, then pass it to any model in the model_* files.

CAUSE NAMING SCHEME
-------------------
Only C causes (C01, C02, ...) actually generate the effect E via noisy-OR.
All other named causes are LURES — they are present in the dataset but have
no causal role in generating E. The L prefix marks lure causes explicitly.

Cause names encode type, timing, and schedule:

  C causes (true causes):
    C01, C02, ...   : main causes; hour and true cs set via C_causes arg

  L causes (lures — do NOT generate E):
    First letter  : L = lure
    Second letter : C = confounded with C01 (same days)
                    U = uncorrelated with C01 (independent, 50/50)
                    S = sporadic (occurs exactly once across all days)
    Third letter  : S = specified time (user provides hour via LCSF_hours)
                    I = interim (between C01 and E)
                    B = before (before C01)
                    A = any (anywhere before E)
    Fourth letter : F = fixed hour (same hour every day present)
                    R = random hour (new random hour each day present)
                    (LS causes have no fourth letter; always random)
    Number        : two-digit index within that type, e.g. 01, 02
    _hour suffix  : appended for fixed causes, e.g. LCIF01_14

  BG : background cause (always present, no specific hour)

  Examples:
    C01          : first main cause
    LCSF01_7     : first lure confound at specified fixed time, hour 7
    LCIF01_14    : first lure confound interim fixed, hour 14
    LCIR01       : first lure confound interim random
    LCBF01_3     : first lure confound before-C fixed, hour 3
    LCBR01       : first lure confound before-C random
    LCAR01       : first lure confound any-time random
    LUIF01_10    : first lure uncorrelated interim fixed, hour 10
    LUIR01       : first lure uncorrelated interim random
    LUBF01_2     : first lure uncorrelated before-C fixed, hour 2
    LUBR01       : first lure uncorrelated before-C random
    LUAR01       : first lure uncorrelated any-time random
    LSIF01       : first lure sporadic interim
    LSBF01       : first lure sporadic before-C
    LSAF01       : first lure sporadic any-time
    BG           : background cause (always present, no specific hour)

DATASET BUNDLE
--------------
make_dataset() returns a dict with keys:
  dataset["day"]            : one row per trial/day
  dataset["hour"]           : one row per clock-hour (24 rows per day),
                              includes absolute_hour column
  dataset["cause_metadata"] : DataFrame describing each cause
  dataset["C01_hour"]       : int, clock hour of C01
  dataset["E_hour"]         : int, clock hour E is observed

Requires: numpy, pandas, scipy, plotnine, matplotlib
"""

import numpy as np
import pandas as pd
import re
from scipy.stats import beta as scipy_beta, gaussian_kde
from plotnine import (
    ggplot, aes, geom_line,
    labs, theme_minimal, theme, element_text,
    scale_color_manual, scale_x_continuous, scale_y_continuous,
    coord_cartesian
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# SECTION 1: CONSTANTS
# =============================================================================

# Scaling constant for the Beta prior on causal strength (Bayesian model).
# cs ~ Beta(1, 1 + K * delta)
# K=1 : prior mean at delta=12 is 1/14 ~ 0.07
# Use plot_strength_prior() to visualise before changing.
K = 1.0

# Number of clock hours in a day
HOURS_PER_DAY = 24


# =============================================================================
# SECTION 2: PRIOR VISUALISATION
# =============================================================================

def plot_strength_prior(delays=None, k_values=None):
    """
    Plot the Beta prior on causal strength for a range of delays and K values.
    Use this to choose K before fitting the Bayesian model.

    Parameters
    ----------
    delays   : list of float, delays to show (default: [0, 6, 12, 24])
    k_values : list of float, K values to compare (default: [0.5, 1.0, 2.0])

    Returns
    -------
    tuple of two plotnine plot objects: (prior_mean_plot, prior_shape_plot)
    """
    if delays is None:
        delays = [0, 6, 12, 24]
    if k_values is None:
        k_values = [0.5, 1.0, 2.0]

    palette     = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#558B2F"]
    delta_range = np.linspace(0, 24, 200)

    rows_mean = []
    for k in k_values:
        label = f"K = {k}" + (" [current]" if k == K else "")
        for d in delta_range:
            rows_mean.append({
                "delta": d,
                "mean" : 1.0 / (2.0 + k * d),
                "k"    : label
            })

    df_mean   = pd.DataFrame(rows_mean)
    labels    = df_mean["k"].unique().tolist()
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}

    p1 = (
        ggplot(df_mean, aes(x="delta", y="mean", color="k"))
        + geom_line(size=1.0)
        + scale_x_continuous(breaks=list(range(0, 25, 4)))
        + scale_y_continuous(limits=(0, 0.55),
                             breaks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
        + scale_color_manual(values=color_map)
        + labs(
            title = "Prior Mean of Causal Strength as a Function of Delay",
            x     = "Delay (hours)",
            y     = "Prior mean of causal strength",
            color = "K value"
        )
        + theme_minimal()
        + theme(
            plot_title   = element_text(size=12, face="bold"),
            axis_title   = element_text(size=10),
            legend_title = element_text(size=9),
            legend_text  = element_text(size=8),
        )
    )
    p1.draw()
    plt.show()

    strength_vals = np.linspace(0.001, 0.999, 300)

    rows_shape = []
    for d in delays:
        b     = 1.0 + K * d
        label = f"delta = {int(d)}h  (Beta(1, {b:.1f}))"
        dens  = scipy_beta.pdf(strength_vals, a=1.0, b=b)
        for s, den in zip(strength_vals, dens):
            rows_shape.append({"strength": s, "density": den, "delay": label})

    df_shape     = pd.DataFrame(rows_shape)
    delay_labels = df_shape["delay"].unique().tolist()
    shape_colors = {lbl: palette[i % len(palette)]
                    for i, lbl in enumerate(delay_labels)}

    p2 = (
        ggplot(df_shape, aes(x="strength", y="density", color="delay"))
        + geom_line(size=1.0)
        + scale_x_continuous(breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + coord_cartesian(xlim=(0, 1))
        + scale_color_manual(values=shape_colors)
        + labs(
            title = f"Prior Shape at Representative Delays  (K = {K})",
            x     = "Causal strength",
            y     = "Prior density",
            color = "Delay"
        )
        + theme_minimal()
        + theme(
            plot_title   = element_text(size=12, face="bold"),
            axis_title   = element_text(size=10),
            legend_title = element_text(size=9),
            legend_text  = element_text(size=8),
        )
    )
    p2.draw()
    plt.show()

    return p1, p2


# =============================================================================
# SECTION 3: DATA GENERATION — INTERNAL HELPERS
# =============================================================================

def _make_cause_metadata(cause_records):
    """
    Build the cause_metadata DataFrame from a list of record dicts.

    Each record has keys: name, type, timing, schedule, hour, true_cs, order.
    true_cs is the ground-truth causal strength used in data generation
    (only set for C causes; NaN for all lures and BG).
    """
    return pd.DataFrame(
        cause_records,
        columns=["name", "type", "timing", "schedule", "hour", "true_cs", "order"]
    )


def _valid_interim_hours(C_hour, E_hour):
    """Return list of valid hours strictly between C_hour and E_hour."""
    return list(range(C_hour + 1, E_hour))


def _valid_before_hours(C_hour):
    """Return list of valid hours strictly before C_hour (hours 1..C_hour-1)."""
    return list(range(1, C_hour))


def _valid_any_hours(C_hour, E_hour):
    """Return list of valid hours strictly before E_hour (hours 1..E_hour-1)."""
    return list(range(1, E_hour))


def _expand_to_hours(df_day, fixed_cause_hours, random_cause_hours_by_day,
                     E_hour, cause_cols):
    """
    Expand a day-level dataframe to an hour-level dataframe.

    Each day becomes HOURS_PER_DAY rows (hours 1 through HOURS_PER_DAY).
    An absolute_hour column is added: absolute_hour = (day - 1) * 24 + hour.
    This allows cross-day interval calculations without reference to days.

    Parameters
    ----------
    df_day                    : pd.DataFrame, day-level data
    fixed_cause_hours         : dict, cause_name -> int hour (fixed causes only)
    random_cause_hours_by_day : dict, cause_name -> np.ndarray of length n_days
    E_hour                    : int, clock hour E is observed
    cause_cols                : list of cause column names (excludes BG)

    Returns
    -------
    pd.DataFrame with columns: day, hour, absolute_hour, <cause_cols>, E
    """
    n_days  = len(df_day)
    n_hours = HOURS_PER_DAY
    n_rows  = n_days * n_hours

    out_day  = np.repeat(df_day["day"].values, n_hours)
    out_hour = np.tile(np.arange(1, n_hours + 1), n_days)
    out_absolute_hour = (out_day - 1) * HOURS_PER_DAY + out_hour

    out_causes = {}
    for col in cause_cols:
        arr          = np.zeros(n_rows, dtype=int)
        present_mask = df_day[col].values == 1

        if col in fixed_cause_hours:
            h = fixed_cause_hours[col]
            for d_idx in np.where(present_mask)[0]:
                arr[d_idx * n_hours + h - 1] = 1

        elif col in random_cause_hours_by_day:
            hours_arr = random_cause_hours_by_day[col]
            for d_idx in np.where(present_mask)[0]:
                h = int(hours_arr[d_idx])
                arr[d_idx * n_hours + h - 1] = 1

        out_causes[col] = arr

    out_E        = np.zeros(n_rows, dtype=int)
    present_mask = df_day["E"].values == 1
    for d_idx in np.where(present_mask)[0]:
        out_E[d_idx * n_hours + E_hour - 1] = 1

    df_hour = pd.DataFrame({
        "day"          : out_day,
        "hour"         : out_hour,
        "absolute_hour": out_absolute_hour
    })
    for col in cause_cols:
        df_hour[col] = out_causes[col]
    df_hour["E"] = out_E

    return df_hour.reset_index(drop=True)


# =============================================================================
# SECTION 4: DATA GENERATION — make_dataset
# =============================================================================

def make_dataset(
    n_days,
    C_causes,
    E_hour,
    LCSF_hours = None,   # list of hours for lure confound specified fixed causes
    LUSF_hours = None,   # list of hours for lure uncorrelated specified fixed causes
    n_LCIF = 0,          # N of lure confound interim fixed
    n_LCIR = 0,          # N of lure confound interim random
    n_LCBF = 0,          # N of lure confound before-C fixed
    n_LCBR = 0,          # N of lure confound before-C random
    n_LCAR = 0,          # N of lure confound any-time random
    n_LUIF = 0,          # N of lure uncorrelated interim fixed
    n_LUIR = 0,          # N of lure uncorrelated interim random
    n_LUBF = 0,          # N of lure uncorrelated before-C fixed
    n_LUBR = 0,          # N of lure uncorrelated before-C random
    n_LUAR = 0,          # N of lure uncorrelated any-time random
    n_LSIF = 0,          # N of lure sporadic interim
    n_LSBF = 0,          # N of lure sporadic before-C
    n_LSAF = 0,          # N of lure sporadic any-time
    seed   = 42
):
    """
    Generate a dataset bundle for causal learning experiments.

    Only C causes (C01, C02, ...) generate E via noisy-OR. All other
    causes are lures (L prefix) — present in the data but causally inert.

    Trials are fully randomized (global shuffle) — not block-randomized.
    The C cause presence schedule uses a full factorial design to guarantee
    exact 50% marginals and exact mutual independence, then all n_days rows
    are shuffled in a single global permutation.

    Parameters
    ----------
    n_days     : total trials; must be divisible by 2^n_C and by 4
    C_causes   : list of (hour, true_cs) tuples, one per main cause;
                 first tuple = C01, second = C02, etc.
    E_hour     : clock hour (1-24) when E is observed; must exceed C01 hour
    LCSF_hours : list of clock hours for lure confound specified fixed causes;
                 each element creates one LCSF cause at that exact hour,
                 confounded with C01 (same days, same schedule)
    LUSF_hours : list of clock hours for lure uncorrelated specified fixed causes;
                 each element creates one LUSF cause at that exact hour,
                 present on 50% of days independent of C01
    n_LCIF     : N of lure confound interim fixed causes
    n_LCIR     : N of lure confound interim random causes
    n_LCBF     : N of lure confound before-C fixed causes
    n_LCBR     : N of lure confound before-C random causes
    n_LCAR     : N of lure confound any-time random causes
    n_LUIF     : N of lure uncorrelated interim fixed causes
    n_LUIR     : N of lure uncorrelated interim random causes
    n_LUBF     : N of lure uncorrelated before-C fixed causes
    n_LUBR     : N of lure uncorrelated before-C random causes
    n_LUAR     : N of lure uncorrelated any-time random causes
    n_LSIF     : N of lure sporadic interim causes
    n_LSBF     : N of lure sporadic before-C causes
    n_LSAF     : N of lure sporadic any-time causes
    seed       : integer random seed for reproducibility (default 42)

    Returns
    -------
    dict with keys:
        "day"            : pd.DataFrame, one row per trial
        "hour"           : pd.DataFrame, one row per (day x clock-hour);
                           includes absolute_hour column
        "cause_metadata" : pd.DataFrame, one row per cause
        "C01_hour"       : int, clock hour of C01
        "E_hour"         : int, clock hour E is observed

    Raises
    ------
    ValueError if n_days not divisible by 2^n_C or 4, E_hour <= C01_hour,
    or no valid hours exist for requested cause types.
    """
    if not C_causes:
        raise ValueError("C_causes must contain at least one (hour, cs) tuple.")

    n_C      = len(C_causes)
    C01_hour = C_causes[0][0]

    required_multiple = 2 ** n_C
    if n_days % required_multiple != 0:
        raise ValueError(
            f"n_days must be divisible by 2^n_C = {required_multiple} "
            f"(got n_days={n_days}, n_C={n_C})."
        )
    if n_days % 4 != 0:
        raise ValueError(
            f"n_days must be divisible by 4 (got n_days={n_days})."
        )
    if E_hour <= C01_hour:
        raise ValueError(
            f"E_hour ({E_hour}) must be strictly greater than C01_hour ({C01_hour})."
        )

    if LCSF_hours is None:
        LCSF_hours = []
    if LUSF_hours is None:
        LUSF_hours = []

    rng = np.random.default_rng(seed)

    interim_hours = _valid_interim_hours(C01_hour, E_hour)
    before_hours  = _valid_before_hours(C01_hour)
    any_hours     = _valid_any_hours(C01_hour, E_hour)

    if (n_LCIF + n_LCIR + n_LUIF + n_LUIR + n_LSIF) > 0 and not interim_hours:
        raise ValueError(
            f"No valid interim hours between C01_hour={C01_hour} and E_hour={E_hour}."
        )
    if (n_LCBF + n_LCBR + n_LUBF + n_LUBR + n_LSBF) > 0 and not before_hours:
        raise ValueError(
            f"No valid before-C01 hours (C01_hour={C01_hour})."
        )
    if (n_LCAR + n_LUAR + n_LSAF) > 0 and not any_hours:
        raise ValueError(
            f"No valid any-time hours before E_hour={E_hour}."
        )

    # ------------------------------------------------------------------
    # C cause presence schedules — full factorial, globally shuffled
    # ------------------------------------------------------------------
    n_per_cell = n_days // required_multiple
    combos = np.array(
        [[int(b) for b in format(k, f"0{n_C}b")]
         for k in range(required_multiple)]
    )
    assignments       = np.repeat(combos, n_per_cell, axis=0)
    shuffle_idx       = rng.permutation(n_days)
    C_presence_matrix = assignments[shuffle_idx]

    # ------------------------------------------------------------------
    # E presence — noisy-OR over C causes only
    # ------------------------------------------------------------------
    C_cs_values = np.array([cs for _, cs in C_causes])
    p_no_effect = np.ones(n_days)
    for ci in range(n_C):
        p_no_effect *= (1.0 - C_cs_values[ci]) ** C_presence_matrix[:, ci]
    p_effect   = 1.0 - p_no_effect
    E_presence = (rng.random(n_days) < p_effect).astype(int)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    cause_records             = []
    fixed_cause_hours         = {}
    random_cause_hours_by_day = {}
    elapsed_by_col            = {}
    presence_by_col           = {}
    cause_col_order           = []

    def _add_cause(name, ctype, timing, schedule, hour_fixed,
                   presence, hours_by_day=None, true_cs=np.nan):
        presence_by_col[name] = presence
        cause_col_order.append(name)
        if hour_fixed is not None:
            fixed_cause_hours[name] = hour_fixed
            elapsed_by_col[name]    = float(E_hour - hour_fixed)
        else:
            random_cause_hours_by_day[name] = hours_by_day
            present_mask = presence == 1
            if present_mask.any():
                elapsed_by_col[name] = float(
                    np.mean(E_hour - hours_by_day[present_mask])
                )
            else:
                elapsed_by_col[name] = float(
                    E_hour - np.mean([any_hours[0], any_hours[-1]])
                )
        order = sum(
            1 for r in cause_records
            if r["type"] == ctype
            and r["timing"] == timing
            and r["schedule"] == schedule
        ) + 1
        cause_records.append({
            "name"    : name,
            "type"    : ctype,
            "timing"  : timing,
            "schedule": schedule,
            "hour"    : hour_fixed,
            "true_cs" : true_cs,
            "order"   : order
        })

    def _confounded_presence():
        return C_presence_matrix[:, 0].copy()

    C01_present_idx = np.where(C_presence_matrix[:, 0] == 1)[0]
    C01_absent_idx  = np.where(C_presence_matrix[:, 0] == 0)[0]
    n_quarter       = n_days // 4

    def _uncorrelated_presence():
        perm_present = rng.permutation(C01_present_idx)
        perm_absent  = rng.permutation(C01_absent_idx)
        u_present    = np.concatenate([perm_present[:n_quarter],
                                       perm_absent[:n_quarter]])
        presence     = np.zeros(n_days, dtype=int)
        presence[u_present] = 1
        return presence

    def _sporadic_presence():
        presence = np.zeros(n_days, dtype=int)
        presence[int(rng.integers(0, n_days))] = 1
        return presence

    def _random_hours_array(valid, n, presence):
        arr = np.zeros(n, dtype=int)
        for d in np.where(presence == 1)[0]:
            arr[d] = int(rng.choice(valid))
        return arr

    # ------------------------------------------------------------------
    # C causes (true causes — generate E)
    # ------------------------------------------------------------------
    for ci, (hour, cs_val) in enumerate(C_causes, start=1):
        name = f"C{ci:02d}"
        _add_cause(name, "C", "", "F", hour,
                   C_presence_matrix[:, ci - 1].copy(),
                   true_cs=cs_val)

    C01_presence = C_presence_matrix[:, 0]

    # ------------------------------------------------------------------
    # L causes (lures — do NOT generate E)
    # ------------------------------------------------------------------

    # LCSF — lure confound specified fixed
    for idx, h in enumerate(LCSF_hours, start=1):
        name = f"LCSF{idx:02d}_{h}"
        _add_cause(name, "LC", "S", "F", h, C01_presence.copy())

    # LUSF — lure uncorrelated specified fixed
    for idx, h in enumerate(LUSF_hours, start=1):
        name = f"LUSF{idx:02d}_{h}"
        _add_cause(name, "LU", "S", "F", h, _uncorrelated_presence())

    # LCIF — lure confound interim fixed
    for i in range(1, n_LCIF + 1):
        h    = int(rng.choice(interim_hours))
        name = f"LCIF{i:02d}_{h}"
        _add_cause(name, "LC", "I", "F", h, _confounded_presence())

    # LCIR — lure confound interim random
    for i in range(1, n_LCIR + 1):
        name = f"LCIR{i:02d}"
        pres = _confounded_presence()
        hrs  = _random_hours_array(interim_hours, n_days, pres)
        _add_cause(name, "LC", "I", "R", None, pres, hrs)

    # LCBF — lure confound before-C fixed
    for i in range(1, n_LCBF + 1):
        h    = int(rng.choice(before_hours))
        name = f"LCBF{i:02d}_{h}"
        _add_cause(name, "LC", "B", "F", h, _confounded_presence())

    # LCBR — lure confound before-C random
    for i in range(1, n_LCBR + 1):
        name = f"LCBR{i:02d}"
        pres = _confounded_presence()
        hrs  = _random_hours_array(before_hours, n_days, pres)
        _add_cause(name, "LC", "B", "R", None, pres, hrs)

    # LCAR — lure confound any-time random
    for i in range(1, n_LCAR + 1):
        name = f"LCAR{i:02d}"
        pres = _confounded_presence()
        hrs  = _random_hours_array(any_hours, n_days, pres)
        _add_cause(name, "LC", "A", "R", None, pres, hrs)

    # LUIF — lure uncorrelated interim fixed
    for i in range(1, n_LUIF + 1):
        h    = int(rng.choice(interim_hours))
        name = f"LUIF{i:02d}_{h}"
        _add_cause(name, "LU", "I", "F", h, _uncorrelated_presence())

    # LUIR — lure uncorrelated interim random
    for i in range(1, n_LUIR + 1):
        name = f"LUIR{i:02d}"
        pres = _uncorrelated_presence()
        hrs  = _random_hours_array(interim_hours, n_days, pres)
        _add_cause(name, "LU", "I", "R", None, pres, hrs)

    # LUBF — lure uncorrelated before-C fixed
    for i in range(1, n_LUBF + 1):
        h    = int(rng.choice(before_hours))
        name = f"LUBF{i:02d}_{h}"
        _add_cause(name, "LU", "B", "F", h, _uncorrelated_presence())

    # LUBR — lure uncorrelated before-C random
    for i in range(1, n_LUBR + 1):
        name = f"LUBR{i:02d}"
        pres = _uncorrelated_presence()
        hrs  = _random_hours_array(before_hours, n_days, pres)
        _add_cause(name, "LU", "B", "R", None, pres, hrs)

    # LUAR — lure uncorrelated any-time random
    for i in range(1, n_LUAR + 1):
        name = f"LUAR{i:02d}"
        pres = _uncorrelated_presence()
        hrs  = _random_hours_array(any_hours, n_days, pres)
        _add_cause(name, "LU", "A", "R", None, pres, hrs)

    # LSIF — lure sporadic interim
    for i in range(1, n_LSIF + 1):
        name = f"LSIF{i:02d}"
        pres = _sporadic_presence()
        hrs  = _random_hours_array(interim_hours, n_days, pres)
        _add_cause(name, "LS", "I", "F", None, pres, hrs)

    # LSBF — lure sporadic before-C
    for i in range(1, n_LSBF + 1):
        name = f"LSBF{i:02d}"
        pres = _sporadic_presence()
        hrs  = _random_hours_array(before_hours, n_days, pres)
        _add_cause(name, "LS", "B", "F", None, pres, hrs)

    # LSAF — lure sporadic any-time
    for i in range(1, n_LSAF + 1):
        name = f"LSAF{i:02d}"
        pres = _sporadic_presence()
        hrs  = _random_hours_array(any_hours, n_days, pres)
        _add_cause(name, "LS", "A", "F", None, pres, hrs)

    # ------------------------------------------------------------------
    # Build day-level DataFrame
    # ------------------------------------------------------------------
    df_day = pd.DataFrame({"day": np.arange(1, n_days + 1)})
    for col in cause_col_order:
        df_day[col] = presence_by_col[col]
    df_day["BG"] = 1
    df_day["E"]  = E_presence
    for col in cause_col_order:
        df_day[f"{col}_elapsed"] = elapsed_by_col[col]

    # ------------------------------------------------------------------
    # Build hour-level DataFrame (includes absolute_hour)
    # ------------------------------------------------------------------
    df_hour = _expand_to_hours(
        df_day, fixed_cause_hours, random_cause_hours_by_day,
        E_hour, cause_cols=cause_col_order
    )
    df_hour["BG"] = 1

    # ------------------------------------------------------------------
    # cause_metadata table
    # ------------------------------------------------------------------
    cause_records.append({
        "name": "BG", "type": "BG", "timing": "",
        "schedule": "", "hour": None, "true_cs": np.nan, "order": 1
    })
    metadata = _make_cause_metadata(cause_records)

    return {
        "day"           : df_day,
        "hour"          : df_hour,
        "cause_metadata": metadata,
        "C01_hour"      : C01_hour,
        "E_hour"        : E_hour
    }


# =============================================================================
# SECTION 5: SHARED UTILITIES
# =============================================================================

def get_cause_columns(dataset):
    """
    Return cause column names in display order from the day-level dataframe.
    C causes first, then all lure types in creation order, then BG last.
    Excludes E and _elapsed columns.
    """
    meta     = {"day", "E"}
    all_cols = [
        c for c in dataset["day"].columns
        if c not in meta and not c.endswith("_elapsed")
    ]
    metadata = dataset["cause_metadata"]
    ordered  = metadata["name"].tolist()
    non_bg   = [c for c in ordered if c != "BG" and c in all_cols]
    has_bg   = "BG" in all_cols
    return non_bg + (["BG"] if has_bg else [])


def check_no_random_causes(dataset):
    """
    Raise ValueError if any random-delay lure causes are present.
    BayesianNoisyOr requires fixed known delays for all causes.
    """
    metadata      = dataset["cause_metadata"]
    random_causes = metadata[metadata["schedule"] == "R"]["name"].tolist()
    if random_causes:
        raise ValueError(
            f"BayesianNoisyOr does not support random-delay causes.\n"
            f"Random-delay causes found: {random_causes}."
        )


def get_elapsed(dataset):
    """
    Return dict mapping cause name -> elapsed time in hours (float).
    BG excluded. Used by BayesianNoisyOr to set delay-weighted priors.
    """
    df_day = dataset["day"]
    result = {}
    for col in get_cause_columns(dataset):
        if col == "BG":
            continue
        elapsed_col = f"{col}_elapsed"
        if elapsed_col in df_day.columns:
            result[col] = float(df_day[elapsed_col].iloc[0])
    return result


def summarise_dataset(dataset):
    """
    Print a human-readable summary of a dataset bundle.
    """
    df_day   = dataset["day"]
    metadata = dataset["cause_metadata"]
    n_days   = len(df_day)
    E_hour   = dataset["E_hour"]
    C01_hour = dataset["C01_hour"]

    print(f"Dataset summary")
    print(f"  n_days   : {n_days}")
    print(f"  C01_hour : {C01_hour}")
    print(f"  E_hour   : {E_hour}")
    print(f"  E rate   : {df_day['E'].mean():.3f}")
    print()
    print(f"  {'Cause':<18} {'Type':<6} {'Timing':<8} {'Sched':<6} "
          f"{'Hour':<6} {'True cs':<10} {'Presence'}")
    print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*10} {'-'*8}")

    cause_cols = get_cause_columns(dataset)
    for col in cause_cols:
        row      = metadata[metadata["name"] == col].iloc[0]
        true_cs  = f"{row['true_cs']:.3f}" if not pd.isna(row["true_cs"]) else "—"
        hour_str = (str(int(row["hour"]))
                    if (row["hour"] is not None and not pd.isna(row["hour"]))
                    else "varies")
        pres     = df_day[col].mean() if col in df_day.columns else float("nan")
        print(f"  {col:<18} {row['type']:<6} {row['timing']:<8} "
              f"{row['schedule']:<6} {hour_str:<6} {true_cs:<10} {pres:.3f}")


# =============================================================================
# SECTION 6: TEMPORAL EXPANSION
# =============================================================================

def expand_to_substeps(dataset, steps_per_hour):
    """
    Expand an hourly dataset to sub-hour resolution by inserting empty
    timesteps within each hour.

    Each clock hour becomes steps_per_hour rows. Events (causes and E)
    fire only at the first substep of their hour (substep 0). All other
    substeps are zeros — nothing happens.

    Hour values become fractional: hour 16 with steps_per_hour=1000
    expands to 16.000, 16.001, 16.002, ..., 16.999. The absolute_hour
    column is recomputed as a float running continuously across all days.

    This is used to model the TD model at a finer timescale — for example,
    comparing a paradigm where events unfold over hours vs. seconds. With
    finer timesteps and the same beta parameter, the eligibility trace
    decays more between events, naturally capturing longer real-time delays.

    Only dataset["hour"] is modified. dataset["day"] and all other keys
    are unchanged, so the expanded dataset can be passed directly to
    run_td_model() or run_gallistel_model(). BayesianNoisyOr is unaffected
    since it only uses dataset["day"].

    Parameters
    ----------
    dataset        : dict, output of make_dataset()
    steps_per_hour : int, number of timesteps per hour (required, no default).
                     Use 60 for minute-level, 1000 for finer resolution.
                     Original hourly data = steps_per_hour of 1.

    Returns
    -------
    dict : a new dataset bundle identical to the input except that
           dataset["hour"] is replaced with the expanded DataFrame.
           The expanded DataFrame has columns:
               day, hour (float), absolute_hour (float), <cause_cols>, BG, E

    Example
    -------
    dataset_hourly = make_dataset(n_days=24, C_causes=[(16, 1.0)],
                                  E_hour=22, ...)
    dataset_minute = expand_to_substeps(dataset=dataset_hourly,
                                        steps_per_hour=60)
    td_minute = run_td_model(dataset=dataset_minute, alpha=0.2,
                             beta=0.8, gamma=1.0, ...)
    """
    import copy

    df_hour    = dataset["hour"].copy()
    cause_cols = get_cause_columns(dataset)   # includes BG
    all_cols   = [c for c in cause_cols] + ["E"]

    n_rows_hour = len(df_hour)

    # Repeat each hour row steps_per_hour times
    df_exp = df_hour.loc[
        df_hour.index.repeat(steps_per_hour)
    ].reset_index(drop=True)

    # Substep index within each hour: 0, 1, ..., steps_per_hour-1
    # Must tile within each original row, not across the whole DataFrame
    substep = np.tile(np.arange(steps_per_hour), n_rows_hour)

    # Zero out events at all substeps except 0 — BG excluded (always 1)
    not_zero    = substep != 0
    cols_to_zero = [c for c in all_cols if c != "BG"]
    for col in cols_to_zero:
        if col in df_exp.columns:
            df_exp.loc[not_zero, col] = 0

    # Fractional hour: original_hour + substep / steps_per_hour
    df_exp["hour"] = df_exp["hour"] + substep / steps_per_hour

    # Recompute absolute_hour as float
    df_exp["absolute_hour"] = (
        (df_exp["day"] - 1) * HOURS_PER_DAY + df_exp["hour"]
    )

    df_exp = df_exp.reset_index(drop=True)

    # Return new dataset dict with expanded hour DataFrame
    new_dataset          = copy.copy(dataset)
    new_dataset["hour"]  = df_exp
    return new_dataset


# =============================================================================
# SECTION 7: COSTA & BOAKES (2011) DATASETS
# =============================================================================
"""
CBEx1 / CBEx2 generate stimulus-only event streams that mimic the design of
Costa & Boakes (2011), "Varying temporal contiguity and interference in a
human avoidance task" (JEP:ABP), Experiments 1 and 2 (the "Martian" task).

WHAT WE ARE / ARE NOT SIMULATING
---------------------------------
We generate STIMULI ONLY. We are NOT simulating any participant behavior:
  - No simulated spacebar presses.
  - No simulated "invasions" (an invasion is a behavioral outcome — it only
    happens if a participant fails to release the spacebar during the
    shield window — so it has no stimulus-level analogue here).
  - No simulated "SAVED" events (also a behavioral outcome).
  - No simulated Martian ships ticking off time. In the original task,
    ships appeared on screen at a roughly constant rate purely to give
    participants a visual clock; that role is now just played directly by
    our regular tick grid (see TICK RESOLUTION below). Ships themselves are
    not represented as a cause or event.
  - No simulated identification-test responses or confidence ratings
    (Experiment 2 blocks).

MAPPING ONTO THE EXISTING C / L / BG SCHEME
--------------------------------------------
  C01  : the warning signal. It deterministically precedes the shield by a
         fixed delay (the "trace interval"), every single trial. This is
         NOT a probabilistic noisy-OR relationship like other C causes in
         this module — the shield always follows the signal. We still
         record true_cs = 1 in cause_metadata to reflect this determinism,
         but no noisy-OR draw is performed.
  L01, L02, ... : distractor symbols (one column per distractor identity).
         We do NOT reuse the LC/LU/LS naming convention (e.g. LUIR, LUBR)
         because that convention assumes a lure's presence is a per-day
         binary that is either confounded with or independent of C01 —
         a "same days / different days" structure that doesn't apply here,
         since distractors fire (or don't) at random ticks within every
         trial, not according to a day-level presence pattern. Plain L01,
         L02, ... labels are used instead.
  E    : the shield onset (NOT the "SAVED" text, which is a behavioral
         outcome we don't simulate). One shield occurs per trial,
         deterministically, trace-interval seconds after the signal.
  BG   : included for consistency with other datasets in this module
         (always 1), even though the original task has no background-cause
         concept.

WHAT A "TRIAL" MEANS HERE
--------------------------
The task is one continuous real-time stream — there is no pause between
trials. A trial is just the label for one signal -> trace interval ->
shield -> ITI cycle; the next trial's signal follows immediately after the
ITI. We do keep a "day" column (one integer per trial) purely because
get_cause_columns() and analysis.py's checkpointing / plotting logic key
off dataset["day"], not because trials are meaningfully day-like. Time
itself is represented as one continuous, ever-increasing column (see below)
that does NOT reset at trial boundaries.

EVENTS ARE INSTANTANEOUS POINTS, NOT DURATIONS
------------------------------------------------
In the original study, signal/shield/distractor symbols were each visible
for ~1-2s. For our purposes we treat all three as instantaneous point
events (a single tick each) rather than events with duration. This avoids
having to model overlap between distractors, or between a distractor and
the signal/shield, and only shifts timing values very slightly relative to
the original paper.

TIME, NOT DAY/HOUR
------------------
Because trials are variable-length and events happen on a sub-second
timescale, we do not use clock-hour-of-day the way make_dataset() does.
Instead we generate a single continuous "time" column, in seconds
(with decimals), that runs across the whole session without resetting at
trial boundaries. This value is ALSO stored in "absolute_hour" so that
existing utilities/models that read dataset["hour"]["absolute_hour"] work
unmodified — but note the units are seconds here, not hours. Model priors
and delay-based constants (e.g. K, used by the Bayesian model) are tuned
for hour-scale delays and will need rescaling before being applied to this
data; that rescaling is deferred for now.

TICK RESOLUTION
----------------
A fixed, regular 250ms tick grid is used for BOTH experiments (Exp 1 and
Exp 2 alike), even though Exp 2's original ship rate was nominally 100ms/
p=.4 per tick (same average rate as Exp 1's 250ms grid, just noisier). We
use 250ms uniformly and adjust the per-tick distractor-firing probability
accordingly (see DISTRACTOR GENERATION below), rather than using a finer
grid for Exp 2 specifically.

TRIAL TIMING
------------
  cycle_length ~ Uniform(low, high)   (signal-onset to next-signal-onset)
    Exp 1: Uniform(20, 30), matching the paper's stated mean 25s, range
           20-30s (the paper does not specify the distributional shape;
           uniform was chosen arbitrarily since it satisfies the mean/range
           given).
    Exp 2: Uniform(17, 27). The paper does not give a range for Exp 2, only
           implied mean ITIs of 19s/16s for trace=3/6s. Assuming the same
           +/-5s spread around the mean as Exp 1, and back-solving the
           implied mean cycle length (~22s), gives this range.
    cycle_length is rounded to the nearest 250ms tick so each trial spans a
    whole number of ticks.
  ITI = cycle_length - trace   (derived, not itself drawn independently)
  Within a trial: signal occurs at trial-relative offset 0 (tick 0);
    shield occurs at trial-relative offset = trace seconds (both trace and
    the tick step are exact multiples of 0.25s for every condition used
    here, so shield always lands exactly on a tick).

DISTRACTOR (LURE) GENERATION
------------------------------
Each of the n_distractors lure symbols is generated INDEPENDENTLY: at every
tick within a trial, each lure symbol has probability 1 / n_ticks_in_trial
of firing at that tick (a low, constant per-tick chance for every lure,
computed from that trial's actual tick count since trial length varies).
This gives each individual lure symbol an expected count of ~1 per trial,
matching the paper's description ("each should occur once on every trial")
for every distractor-count condition, including the 1-distractor condition
(Table 1 in the paper confirms this is genuinely a random count with
nonzero SD even when n_distractors=1, not a forced exact-one occurrence).
This matches the paper's stated pooled formula
(P(any distractor at a tick) = n_distractors / n_ticks_total) when treated
as the sum of n_distractors independent low-probability events. Because
events are instantaneous points (see above), we do not need the paper's
overlap-avoidance adjustment (preventing a new distractor while another
symbol is still visible) — a lure may in principle land on the same tick
as the signal or shield; this rare coincidence is not treated specially.

KNOWN LIMITATIONS / DEFERRED ISSUES
--------------------------------------
  1. BayesianNoisyOr (model_Bayesian_noisy_or.py) will currently refuse to
     run on this data: check_no_random_causes() raises because our lures
     have schedule="R" (they do not have a single fixed delay), and
     get_elapsed() has no well-defined single elapsed time for a lure with
     random per-trial timing. Fixing this requires rethinking the model's
     delay-weighted-prior logic for genuinely stochastic-timing causes; not
     done here.
  2. analysis.py's learning-curve support for the TD model assumes each
     "day" occupies exactly HOURS_PER_DAY (24) rows of dataset["hour"]
     (see _run_learning_curve() and plot_td_weights()'s day-boundary
     lines). Our trials have a variable number of ticks, so:
       - a full, single run of run_td_model() (n_blocks=1) is fine — it
         simply iterates every row in sequence and returns the true final
         weights.
       - learning curves (n_blocks > 1) and the day-boundary reference
         lines drawn on TD plots may be misaligned, since checkpoints are
         computed assuming a fixed 24-tick trial length. CBEx1_condition
         and CBEx2_condition print a warning about this each time they are
         called, rather than silently producing possibly-misleading plots.
     This is not fixed here; only flagged.
  3. C01_hour and E_hour are ordinarily numeric clock hours; here they are
     descriptive strings ("signal_onset" / "trace{trace}") since only
     summarise_dataset() reads them (purely for display) and there is no
     single numeric "hour" concept in this paradigm.
"""

# Fixed tick resolution (seconds) used for both CBEx1 and CBEx2.
CB_TICK_SECONDS = 0.25


def _cb_learning_curve_warning():
    print(
        "  NOTE: this dataset has variable-length trials (see module "
        "docstring, SECTION 7). A single full run (n_blocks=1) of any "
        "model is fine, but analysis.py learning curves (n_blocks > 1) "
        "and TD day-boundary plot lines assume a fixed HOURS_PER_DAY "
        "ticks/trial and may be misaligned for this data."
    )


def _cb_cause_metadata(n_distractors):
    """
    Build the cause_metadata table for a Costa & Boakes dataset.

    C01   : type="C",  true_cs=1 (deterministic signal->shield delay,
            no noisy-OR draw is actually performed)
    L01.. : type="L",  timing="A", schedule="R" (reusing the existing
            timing/schedule codes loosely — these lures do not have a
            day-level present/absent structure the way other L causes in
            this module do, but "any time, random" is the closest fit)
    BG    : included for consistency with other datasets in this module

    Parameters
    ----------
    n_distractors : int, number of distractor (lure) symbols

    Returns
    -------
    pd.DataFrame, same schema as _make_cause_metadata() elsewhere in this
    module: columns [name, type, timing, schedule, hour, true_cs, order]
    """
    records = [{
        "name": "C01", "type": "C", "timing": np.nan, "schedule": np.nan,
        "hour": np.nan, "true_cs": 1, "order": 1
    }]
    for i in range(1, n_distractors + 1):
        records.append({
            "name": f"L{i:02d}", "type": "L", "timing": "A",
            "schedule": "R", "hour": np.nan, "true_cs": np.nan, "order": i
        })
    records.append({
        "name": "BG", "type": "BG", "timing": np.nan, "schedule": np.nan,
        "hour": np.nan, "true_cs": np.nan, "order": 1
    })
    return _make_cause_metadata(records)


def _make_cb_dataset(trace, n_distractors, n_trials,
                     cycle_low, cycle_high, seed,
                     tick=CB_TICK_SECONDS):
    """
    Core generator shared by CBEx1_condition() and CBEx2_condition().

    Builds a continuous-time, tick-level event stream of n_trials
    signal -> trace interval -> shield cycles, each preceded by a randomly
    drawn ITI, with independently-generated distractor (lure) ticks.
    See the SECTION 7 module docstring above for full details.

    Parameters
    ----------
    trace       : float, trace interval in seconds (signal-to-shield delay)
    n_distractors : int, number of distinct distractor symbols
    n_trials    : int, number of signal->shield cycles to generate
    cycle_low   : float, lower bound of Uniform draw for cycle length (s)
    cycle_high  : float, upper bound of Uniform draw for cycle length (s)
    seed        : int, random seed
    tick        : float, tick resolution in seconds (default: CB_TICK_SECONDS)

    Returns
    -------
    dict with keys:
        "day"            : pd.DataFrame, one row per trial (0/1 presence
                           of C01, each L0i, BG, E)
        "hour"           : pd.DataFrame, one row per 250ms tick, columns
                           [day, time, absolute_hour, C01, L01.., BG, E].
                           "time" and "absolute_hour" hold the same value,
                           in seconds, running continuously across the
                           whole session (see module docstring).
        "cause_metadata" : pd.DataFrame (see _cb_cause_metadata)
        "C01_hour"       : str, "signal_onset" (descriptive, not numeric)
        "E_hour"         : str, f"trace{trace}" (descriptive, not numeric)
        "trace"          : float, the trace interval used (seconds)
        "n_distractors"  : int, number of distractor symbols used
    """
    rng       = np.random.default_rng(seed)
    lure_cols = [f"L{i:02d}" for i in range(1, n_distractors + 1)]

    day_rows  = []
    hour_cols = {"day": [], "time": [], "absolute_hour": [],
                 "C01": [], "BG": [], "E": []}
    for c in lure_cols:
        hour_cols[c] = []

    current_time = 0.0

    for trial in range(1, n_trials + 1):
        cycle_length_raw = rng.uniform(cycle_low, cycle_high)
        n_ticks_trial     = max(1, int(round(cycle_length_raw / tick)))
        cycle_length      = n_ticks_trial * tick  # snapped to tick grid

        shield_tick_idx = int(round(trace / tick))
        lure_prob        = 1.0 / n_ticks_trial

        lure_present_trial = {c: 0 for c in lure_cols}

        for tick_idx in range(n_ticks_trial):
            t = current_time + tick_idx * tick

            hour_cols["day"].append(trial)
            hour_cols["time"].append(t)
            hour_cols["absolute_hour"].append(t)
            hour_cols["C01"].append(1 if tick_idx == 0 else 0)
            hour_cols["E"].append(1 if tick_idx == shield_tick_idx else 0)
            hour_cols["BG"].append(1)

            for c in lure_cols:
                fired = int(rng.random() < lure_prob)
                hour_cols[c].append(fired)
                if fired:
                    lure_present_trial[c] = 1

        day_row = {"day": trial, "C01": 1, "E": 1, "BG": 1}
        day_row.update(lure_present_trial)
        day_rows.append(day_row)

        current_time += cycle_length

    hour_col_order = ["day", "time", "absolute_hour", "C01"] + lure_cols + ["BG", "E"]
    df_hour        = pd.DataFrame(hour_cols)[hour_col_order]

    day_col_order = ["day", "C01"] + lure_cols + ["BG", "E"]
    df_day        = pd.DataFrame(day_rows)[day_col_order]

    metadata = _cb_cause_metadata(n_distractors)

    return {
        "day"           : df_day,
        "hour"          : df_hour,
        "cause_metadata": metadata,
        "C01_hour"      : "signal_onset",
        "E_hour"        : f"trace{trace:g}",
        "trace"         : trace,
        "n_distractors" : n_distractors
    }


def CBEx1_condition(trace, n_distractors, n_trials=50, seed=42):
    """
    Generate one Costa & Boakes (2011) Experiment 1 condition.

    Experiment 1 design: 3 (trace: 2, 5, 8s) x 3 (n_distractors: 1, 3, 5),
    between-subjects, 50 trials/participant (paper default).
    cycle_length ~ Uniform(20, 30) seconds (paper: mean 25s, range 20-30s).

    Parameters
    ----------
    trace         : float, trace interval in seconds (e.g. 2, 5, or 8)
    n_distractors : int, number of distinct distractor symbols (e.g. 1, 3, 5)
    n_trials      : int, number of trials to generate (default 50, matching
                    the paper; always pass explicitly per project convention)
    seed          : int, random seed (default 42; always pass explicitly)

    Returns
    -------
    dict : dataset bundle, see _make_cb_dataset() for keys.
    """
    _cb_learning_curve_warning()
    return _make_cb_dataset(
        trace=trace, n_distractors=n_distractors, n_trials=n_trials,
        cycle_low=20, cycle_high=30, seed=seed
    )


def CBEx2_condition(trace, n_distractors, n_trials=75, seed=42):
    """
    Generate one Costa & Boakes (2011) Experiment 2 condition.

    Experiment 2 design: 2 (trace: 3, 6s) x 3 (n_distractors: 1, 3, 5),
    between-subjects, 75 trials/participant (paper default; originally
    split into 3 blocks of 25 for identification tests, which we do not
    simulate — see module docstring).
    cycle_length ~ Uniform(17, 27) seconds. The paper only states implied
    mean ITIs (19s/16s for trace=3/6s); this range is inferred by assuming
    the same +/-5s spread as Experiment 1 around the implied mean cycle
    length of ~22s (see module docstring for the derivation).

    Parameters
    ----------
    trace         : float, trace interval in seconds (e.g. 3 or 6)
    n_distractors : int, number of distinct distractor symbols (e.g. 1, 3, 5)
    n_trials      : int, number of trials to generate (default 75, matching
                    the paper; always pass explicitly per project convention)
    seed          : int, random seed (default 42; always pass explicitly)

    Returns
    -------
    dict : dataset bundle, see _make_cb_dataset() for keys.
    """
    _cb_learning_curve_warning()
    return _make_cb_dataset(
        trace=trace, n_distractors=n_distractors, n_trials=n_trials,
        cycle_low=17, cycle_high=27, seed=seed
    )


def CBEx1(n_trials=50, seed=42):
    """
    Generate all 9 conditions of Costa & Boakes (2011) Experiment 1
    (3 trace values x 3 distractor counts).

    Each condition gets a distinct seed (seed + condition index) so
    conditions are not identical draws of each other.

    Parameters
    ----------
    n_trials : int, number of trials per condition (default 50, matching
               the paper; always pass explicitly per project convention)
    seed     : int, base random seed (default 42; always pass explicitly)

    Returns
    -------
    dict : keys are (trace, n_distractors) tuples, values are dataset
           bundles (see _make_cb_dataset() for keys within each bundle).
    """
    traces        = [2, 5, 8]
    distractor_ns = [1, 3, 5]

    results = {}
    idx     = 0
    for trace in traces:
        for n_distractors in distractor_ns:
            results[(trace, n_distractors)] = CBEx1_condition(
                trace=trace, n_distractors=n_distractors,
                n_trials=n_trials, seed=seed + idx
            )
            idx += 1
    return results


def CBEx2(n_trials=75, seed=42):
    """
    Generate all 6 conditions of Costa & Boakes (2011) Experiment 2
    (2 trace values x 3 distractor counts).

    Each condition gets a distinct seed (seed + condition index) so
    conditions are not identical draws of each other.

    Parameters
    ----------
    n_trials : int, number of trials per condition (default 75, matching
               the paper; always pass explicitly per project convention)
    seed     : int, base random seed (default 42; always pass explicitly)

    Returns
    -------
    dict : keys are (trace, n_distractors) tuples, values are dataset
           bundles (see _make_cb_dataset() for keys within each bundle).
    """
    traces        = [3, 6]
    distractor_ns = [1, 3, 5]

    results = {}
    idx     = 0
    for trace in traces:
        for n_distractors in distractor_ns:
            results[(trace, n_distractors)] = CBEx2_condition(
                trace=trace, n_distractors=n_distractors,
                n_trials=n_trials, seed=seed + idx
            )
            idx += 1
    return results