# Causal Learning Models

Bayesian and associative learning models for estimating causal strength and
cause-effect delay distributions from temporal data. Designed for a
paradigm in which participants observe causes and effects across repeated
daily trials, and the researcher wants an ideal observer model to compare
against human judgments.

---

## Project Overview

The experiment presents participants with two observed causes (C01, C02)
and one effect (E) across 24 daily trials. Each cause and effect occurs at
most once per day, recorded as an hour-of-day timestamp. The models below
estimate causal strength and (for some models) the delay distribution between
each cause and the effect.

---

## File Structure

| File | Purpose |
|---|---|
| `data_generation_and_utils.py` | Dataset generation and shared utilities |
| `model_Bayesian_noisy_or.py` | Bayesian noisy-OR model (Cheng 1997) |
| `model_td.py` | Temporal-difference model (Sutton & Barto 1987) |
| `model_event_based.py` | GPGB rate-based and event-based models (Gong et al. 2025) |
| `model_Gallistel_informativeness.py` | GPI and GRI informativeness measures (Gallistel et al.) |
| `analysis.py` | Multi-dataset runner, learning curves, model comparison |
| `tests.qmd` | Quarto test document with example runs and outputs |
| `README.md` | This file |

All model files import from `data_generation_and_utils.py`. All files must
be in the same directory.

---

## Environment Setup

### Requirements

- Python 3.11
- conda (via Miniforge or Anaconda)
- Quarto 1.9+ (for running `tests.qmd`)

### Creating the conda environment

```bash
conda create -n causal_model python=3.11
conda activate causal_model
conda install -c conda-forge pymc=5.10 arviz=0.18 numpy pandas pytensor
pip install plotnine --break-system-packages
conda install jupyter
```

### Verifying the environment

```bash
conda activate causal_model
python -c "import pymc; import arviz; import plotnine; print('All packages OK')"
python -c "import jupyter; print('Jupyter OK')"
```

### Selecting the interpreter in Positron

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type `Python: Select Interpreter`
3. Choose the interpreter containing `causal_model` in its path
   (typically something like `/Users/yourname/miniforge3/envs/causal_model/bin/python`)

---

## Quick Start

Open Positron, select the `causal_model` interpreter, and run these
sections interactively:

```python
from data_generation_and_utils import make_dataset
from analysis import run_analysis

# Generate a dataset with one main cause (C01) at hour 14, true cs = 0.8,
# and one uncorrelated fixed cause (UIF) at a random interim hour
results = run_analysis(
    dataset_kwargs = dict(
        n_days                    = 24,
        C_causes                  = [(14, 0.8)],
        E_hour                    = 22,
        confounds_specified_times = [],
        n_CIF = 0, n_CIR = 0, n_CBF = 0, n_CBR = 0, n_CAR = 0,
        n_UIF = 1, n_UIR = 0, n_UBF = 0, n_UBR = 0, n_UAR = 0,
        n_SIF = 0, n_SBF = 0, n_SAF = 0,
        seed  = 42
    ),
    models        = ["BayesianNoisyOr", "TD", "Gallistel"],
    n_datasets    = 1,
    n_blocks      = 1,
    draws         = 2000,
    tune          = 1000,
    chains        = 4,
    target_accept = 0.9,
    random_seed   = 42,
    show_plots    = True,
    verbose       = True
)

# Access results
print(results["aggregated_results"]["mean_cs_final"])
```

---

## Models

### BayesianNoisyOr

Estimates causal strength via noisy-OR (Cheng 1997) with delay-weighted
Beta priors. Delays are fixed and known; only causal strength is inferred.
Causes with random delays (CIR, UIR, etc.) are not supported.

```python
from model_Bayesian_noisy_or import run_noisyor_model

bayes_results = run_noisyor_model(
    dataset       = dataset,
    draws         = 2000,
    tune          = 1000,
    chains        = 4,
    target_accept = 0.9,
    random_seed   = 42,
    show_plots    = True,
    verbose       = True
)

bayes_results["summary"]    # posterior summary DataFrame
bayes_results["diag"]       # MCMC diagnostics
bayes_results["idata"]      # ArviZ InferenceData
bayes_results["plot"]       # matplotlib Figure
```

### TD (Temporal Difference)

Associative learning model (Sutton & Barto 1987) operating hour-by-hour.
Updates weights via TD prediction errors and eligibility traces. BG is
excluded (it absorbs credit from real causes in this formulation).

```python
from model_td import run_td_model

td_results = run_td_model(
    dataset    = dataset,
    alpha      = 0.2,   # learning rate
    beta       = 0.8,   # trace decay (higher = longer memory)
    gamma      = 1.0,   # discount factor
    show_plots = True,
    verbose    = True
)

td_results["w_final"]    # final weights, shape (n_causes,)
td_results["w_history"]  # full trajectory, shape (n_steps+1, n_causes)
```

### GPGB Rate-Based Model

Based on Gong, Pacer, Griffiths & Bramley (2025). Each cause contributes
to a Poisson rate at each clock-hour via a normalized gamma delay density.
Infers both causal strength (cs) and delay distribution (shape, rate) for
each cause. Supports random-delay causes.

```python
from model_event_based import run_gpgb_model

gpgb_results = run_gpgb_model(
    dataset       = dataset,
    draws         = 2000,
    tune          = 1000,
    chains        = 4,
    target_accept = 0.9,
    random_seed   = 42,
    max_lag       = 23,
    show_plots    = True,
    verbose       = True
)
```

### Event-Based Model

Also from Gong et al. (2025). Treats all days as a single continuous
timeline and matches cause tokens to effect events via causal pathways.
Two inference modes: `importance_sampling` (structure comparison) or
`nuts` (parameter estimation).

Note: importance sampling is appropriate for model comparison (marginal
likelihoods) but not for sequential updating — ESS collapses after many
trials. Use `nuts` for parameter estimation.

```python
from model_event_based import run_event_based_model

eb_results = run_event_based_model(
    dataset        = dataset,
    inference      = "importance_sampling",
    m              = 10000,
    max_delay      = 48.0,
    max_mean_delay = 100.0,
    override_n     = False,
    random_seed    = 42,
    verbose        = True
)
```

### Gallistel Informativeness (GPI / GRI)

Non-parametric measures based on Gallistel et al. (working paper),
equations 1a and 1b. Computed entirely from temporal statistics on the
continuous (absolute-hour) timeline — trial structure is ignored.

- **GPI** (Prospective): mean inter-E interval / mean C-to-next-E interval
- **GRI** (Retrospective): mean inter-C interval / mean E-to-prior-C interval

Values > 1 indicate the cause is temporally informative about the effect.

```python
from model_Gallistel_informativeness import run_gallistel_model

gal_results = run_gallistel_model(
    dataset = dataset,
    verbose = True
)

gal_results["GPI"]   # dict: cause_name -> GPI value
gal_results["GRI"]   # dict: cause_name -> GRI value
```

---

## Dataset Generation

```python
from data_generation_and_utils import make_dataset

dataset = make_dataset(
    n_days     = 24,            # must be divisible by 2^n_C and by 4
    C_causes   = [(14, 0.8)],   # list of (hour, true_cs); first = C01
    E_hour     = 22,            # clock hour E is observed; must exceed C01 hour
    LCSF_hours = [],            # hours for lure confound specified fixed causes
    LUSF_hours = [],            # hours for lure uncorrelated specified fixed causes
    n_LCIF = 0,                 # N of lure confound interim fixed
    n_LCIR = 0,                 # N of lure confound interim random
    n_LCBF = 0,                 # N of lure confound before-C fixed
    n_LCBR = 0,                 # N of lure confound before-C random
    n_LCAR = 0,                 # N of lure confound any-time random
    n_LUIF = 0,                 # N of lure uncorrelated interim fixed
    n_LUIR = 0,                 # N of lure uncorrelated interim random
    n_LUBF = 0,                 # N of lure uncorrelated before-C fixed
    n_LUBR = 0,                 # N of lure uncorrelated before-C random
    n_LUAR = 0,                 # N of lure uncorrelated any-time random
    n_LSIF = 0,                 # N of lure sporadic interim
    n_LSBF = 0,                 # N of lure sporadic before-C
    n_LSAF = 0,                 # N of lure sporadic any-time
    seed   = 42                 # integer random seed for reproducibility
)

# dataset["day"]            : day-level DataFrame
# dataset["hour"]           : hour-level DataFrame (includes absolute_hour)
# dataset["cause_metadata"] : cause info table
# dataset["C01_hour"]       : int, clock hour of C01
# dataset["E_hour"]         : int, clock hour of E
```

### Cause naming scheme

Cause names encode type, timing, and schedule:

```
L prefix      : all lure causes (do not generate E)
Second letter : C = confounded with C01 (same days)
                U = uncorrelated with C01 (independent, 50% of days)
                S = sporadic (occurs exactly once)
Third letter  : S = specified hour (user provides via LCSF_hours / LUSF_hours)
                I = interim (between C01 and E)
                B = before (before C01)
                A = any (anywhere before E)
Fourth letter : F = fixed hour | R = random hour
Number        : two-digit index (01, 02, ...)
_hour suffix  : appended for fixed causes (e.g. LCIF01_14, LUSF01_18)
```

---

## Learning Curves

Use `n_blocks` in `run_analysis()` to get parameter estimates at intermediate
checkpoints. `n_blocks=4` with 24 trials gives estimates after trials 6, 12,
18, and 24.

```python
results = run_analysis(
    dataset_kwargs = dict(n_days=24, C_causes=[(14, 0.8)],
                          E_hour=22, seed=42,
                          # ... all other params set to 0
                          n_CIF=0, n_CIR=0, n_CBF=0, n_CBR=0, n_CAR=0,
                          n_UIF=0, n_UIR=0, n_UBF=0, n_UBR=0, n_UAR=0,
                          n_SIF=0, n_SBF=0, n_SAF=0,
                          confounds_specified_times=[]),
    models        = ["BayesianNoisyOr", "TD"],
    n_datasets    = 5,
    n_blocks      = 4,
    draws         = 2000,
    tune          = 1000,
    chains        = 4,
    target_accept = 0.9,
    random_seed   = 42,
    show_plots    = False,
    verbose       = True
)

# Aggregated mean cs_C01 at each checkpoint, averaged across 5 datasets:
agg = results["aggregated_results"]
for m in ["BayesianNoisyOr", "TD"]:
    print(f"\n{m}:")
    for i, cp in enumerate(agg["checkpoints"]):
        cs = agg["mean_cs_by_checkpoint"][m][i].get("C01", float("nan"))
        print(f"  After trial {cp}: cs_C01 = {cs:.3f}")
```

---

## Running the Tests

Tests and example outputs are in `tests.qmd`. To render:

```bash
conda activate causal_model
quarto render tests.qmd
```

This produces `tests.html` with all outputs and plots embedded. You can
also run individual code chunks interactively in Positron by opening
`tests.qmd` and using the Run Cell button.

---

## References

- Cheng, P. W. (1997). From covariation to causation: A causal power theory.
  *Psychological Review*, 104(2), 367–405.
- Sutton, R. S., & Barto, A. G. (1987). A temporal-difference model of
  classical conditioning. *Proceedings of the 9th Annual Conference of the
  Cognitive Science Society*.
- Ludvig, E. A., Sutton, R. S., & Kehoe, E. J. (2012). Evaluating the TD
  model of classical conditioning. *Learning & Behavior*, 40(3), 305–319.
- Gong, T., Pacer, M., Griffiths, T. L., & Bramley, N. R. (2025). Rational
  causal induction from time. *Psychological Review*.
- Gallistel, C. R., et al. (working paper). A simple, transparent and useable
  formalization of associative learning, rooted in information theory.
