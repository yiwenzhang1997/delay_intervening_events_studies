"""
model_td.py
===========
Temporal-Difference (TD) model of associative learning.

Implements the TD model from Sutton & Barto (1987) as applied to classical
conditioning, following the parameter conventions of Ludvig et al. (2012).

The model processes each clock-hour of each day in sequence, updating
associative weights via TD prediction errors and eligibility traces.

NOTE: BG (background) is excluded from the TD model. BG has no specific
firing hour and is always present; including it causes its weight to dominate
and absorb credit from real causes. Neither Sutton & Barto (1987) nor
Ludvig et al. (2012) include a background cue in their TD formulation.

MODEL EQUATIONS
---------------
At each time step t:

  Eligibility trace (one per cause i):
    T_i(t) = beta * T_i(t-1) + (1 - beta) * x_i(t-1)
    x_i(t-1) is cause i's presence at the PREVIOUS step.
    Initialized to zero at t=0.

  US prediction (rectified):
    V(t) = max(0, w(t) · x(t))

  TD prediction error:
    delta(t) = E(t) + gamma * V(t) - V(t-1)

  Weight update:
    w_i(t+1) = w_i(t) + alpha * delta(t) * T_i(t)

PARAMETER NAMES
---------------
Following Ludvig et al. (2012):
  alpha : learning rate          (Sutton & Barto: c)
  beta  : trace decay parameter  (higher = longer eligibility memory)
  gamma : discount factor        (1.0 = no discounting)

PRIMARY INTERFACE
-----------------
  results = run_td_model(dataset, alpha, beta, gamma,
                         show_plots=True, verbose=True)

  results["w_final"]    : np.ndarray (n_causes,), final weights
  results["w_history"]  : np.ndarray (n_steps+1, n_causes), full trajectory
  results["cause_cols"] : list of cause column names (BG excluded)
  results["df_hour"]    : hour-level DataFrame used as input
  results["params"]     : dict {alpha, beta, gamma}
  results["plot"]       : matplotlib Figure

References
----------
Sutton, R. S., & Barto, A. G. (1987). A temporal-difference model of
  classical conditioning. Proceedings of the 9th Annual Conference of the
  Cognitive Science Society.

Ludvig, E. A., Sutton, R. S., & Kehoe, E. J. (2012). Evaluating the TD
  model of classical conditioning. Learning & Behavior, 40(3), 305-319.

Requires: numpy, pandas, matplotlib, re
          data_generation_and_utils (same directory)
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from data_generation_and_utils import (
    HOURS_PER_DAY,
    get_cause_columns,
)


# =============================================================================
# SECTION 1: MODEL
# =============================================================================

def run_td_model(dataset, alpha, beta, gamma,
                 show_plots=True, verbose=True):
    """
    Run the TD model on a dataset.

    All model parameters are required (no defaults) to encourage explicit calls.

    Parameters
    ----------
    dataset    : dict, output of make_dataset()
    alpha      : float, learning rate (> 0)
    beta       : float, trace decay parameter (0 < beta < 1;
                 higher = longer eligibility memory)
    gamma      : float, discount factor (0 < gamma <= 1;
                 1.0 = no discounting)
    show_plots : bool, display weight trajectory plot (default True)
    verbose    : bool, print final weights and parameters (default True)

    Returns
    -------
    dict with keys:
        "w_final"   : np.ndarray, shape (n_causes,).
                      Final associative weights after all time steps.
        "w_history" : np.ndarray, shape (n_steps + 1, n_causes).
                      Full weight trajectory. Row 0 is all zeros (the
                      pre-experiment initialization). Row t (t >= 1) holds
                      weights after processing the t-th time step.
        "cause_cols": list of str, cause column names (BG excluded).
        "df_hour"   : pd.DataFrame, the hour-level data used as input.
        "params"    : dict, {alpha, beta, gamma}.
        "plot"      : matplotlib Figure of weight trajectories.
    """
    df_hour    = dataset["hour"]
    cause_cols = [c for c in get_cause_columns(dataset) if c != "BG"]
    n_causes   = len(cause_cols)
    n_steps    = len(df_hour)

    # Read data into arrays once — avoids repeated DataFrame indexing
    X = df_hour[cause_cols].values.astype(float)  # (n_steps, n_causes)
    R = df_hour["E"].values.astype(float)          # (n_steps,)

    # ------------------------------------------------------------------
    # State vectors — overwritten at every time step
    # ------------------------------------------------------------------
    w      = np.zeros(n_causes)   # associative weights
    T      = np.zeros(n_causes)   # eligibility traces
    V_prev = 0.0                  # rectified prediction from previous step

    # ------------------------------------------------------------------
    # Weight history — shape (n_steps + 1, n_causes)
    # Row 0: all zeros (pre-experiment initialization)
    # Row t (t >= 1): weights after processing time step t
    # ------------------------------------------------------------------
    w_history = np.zeros((n_steps + 1, n_causes))

    for t in range(n_steps):

        x_t = X[t]
        r_t = R[t]

        # Rectified US prediction
        V_curr = max(0.0, float(np.dot(w, x_t)))

        # TD prediction error
        delta = r_t + gamma * V_curr - V_prev

        # Update eligibility traces using the PREVIOUS step's stimuli.
        # When t=0, Python's X[t-1] = X[-1] (last row), which is wrong.
        # Explicitly zero-initialize to represent no prior stimuli.
        x_prev = X[t - 1] if t > 0 else np.zeros(n_causes)
        T      = beta * T + (1.0 - beta) * x_prev

        # Weight update
        w = w + alpha * delta * T

        # Record
        w_history[t + 1] = w.copy()

        V_prev = V_curr

    if verbose:
        print("\n=== TD Model — Final Weights ===")
        for col, weight in zip(cause_cols, w):
            print(f"  w_{col} = {weight:.4f}")
        print(f"\n  Parameters: alpha={alpha}, beta={beta}, gamma={gamma}")

    if show_plots:
        fig = plot_td_weights(
            w_history  = w_history,
            cause_cols = cause_cols,
            dataset    = dataset,
            show       = True
        )
    else:
        fig = None

    return {
        "w_final"   : w,
        "w_history" : w_history,
        "cause_cols": cause_cols,
        "df_hour"   : df_hour,
        "params"    : {"alpha": alpha, "beta": beta, "gamma": gamma},
        "plot"      : fig
    }


# =============================================================================
# SECTION 2: PLOTS
# =============================================================================

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
    if type_key == "C_only":
        return [c for c in cause_cols if re.match(r"^C\d{2}$", c)]
    if type_key == "BG_only":
        return [c for c in cause_cols if c == "BG"]
    return [c for c in cause_cols
            if not re.match(r"^C\d{2}$", c) and c != "BG"
            and c.startswith(type_key)]


def plot_td_weights(w_history, cause_cols, dataset, show=True):
    """
    Plot TD associative weight trajectories grouped by cause type.

    Layout: 2 columns wide, as many rows as needed.
    Each non-empty cause type gets one panel with overlaid weight trajectories,
    one line per cause. All panels share the x-axis (hours across experiment).
    Vertical dashed lines mark day boundaries.

    Parameters
    ----------
    w_history  : np.ndarray, shape (n_steps + 1, n_causes).
                 Row 0 = pre-experiment zeros; row t = weights after step t.
    cause_cols : list of str, cause column names in display order (BG excluded)
    dataset    : dict, dataset bundle (used for number of days)
    show       : bool, if True call plt.show() (default True)

    Returns
    -------
    matplotlib Figure
    """
    n_steps   = w_history.shape[0] - 1
    time_axis = np.arange(0, n_steps + 1)
    n_days    = int(dataset["day"]["day"].max())

    panels = []
    for type_key, title in _TYPE_ORDER:
        cols = _causes_of_type(type_key, cause_cols)
        if cols:
            panels.append((type_key, title, cols))

    n_panels = len(panels)
    N_COLS   = 2
    n_rows   = (n_panels + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(nrows=n_rows, ncols=N_COLS,
                             figsize=(13, 4 * n_rows),
                             sharex=True)
    axes = np.array(axes).reshape(n_rows, N_COLS)

    def _add_day_lines(ax):
        for d in range(1, n_days):
            ax.axvline(x=d * HOURS_PER_DAY, color="grey",
                       linewidth=0.5, linestyle="--", alpha=0.4)

    for idx, (type_key, title, cols) in enumerate(panels):
        row = idx // N_COLS
        col = idx % N_COLS
        ax  = axes[row, col]

        _add_day_lines(ax)
        for j, cause in enumerate(cols):
            c_idx = cause_cols.index(cause)
            ax.plot(time_axis, w_history[:, c_idx],
                    color=_PALETTE[j % len(_PALETTE)],
                    linewidth=1.5, label=cause)
        ax.axhline(y=0, color="black", linewidth=0.6, linestyle="-", alpha=0.3)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("Weight  w", fontsize=9)
        if len(cols) <= 10:
            ax.legend(fontsize=7, ncol=1, loc="upper left")

    # Hide unused panels
    for empty_idx in range(n_panels, n_rows * N_COLS):
        axes[empty_idx // N_COLS, empty_idx % N_COLS].set_visible(False)

    # x-axis label on bottom visible panels only
    for col in range(N_COLS):
        last_row = None
        for row in range(n_rows - 1, -1, -1):
            if axes[row, col].get_visible():
                last_row = row
                break
        if last_row is not None:
            axes[last_row, col].set_xlabel(
                "Time step (hours across experiment)", fontsize=9
            )

    axes[0, 0].set_xlim(0, n_steps)

    fig.suptitle("TD Model — Weight Trajectories",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()
    return fig
