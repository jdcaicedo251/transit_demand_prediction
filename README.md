# Open-source codebase for short-term ridership prediction

Models Implemented

- ARIMA
- SARIMA
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Long-Short-Term memory (LSTM)

We kindly request that users of this open-source platform cite this paper in any resulting publications or works that make use of the platform: 

Caicedo, J. D., Guirado, C., González, M. C., & Walker, J. L. (2025). Sharing, collaborating, and benchmarking to advance travel demand research: A demonstration of short-term ridership prediction. Transport Policy, 171, 531–541. https://doi.org/https://doi.org/10.1016/j.tranpol.2025.06.009

## Overview

This project offers code and notebooks for forecasting transit ridership. It supports both **static** training and **online** updates as new data arrives.

### Repository Layout

```
.
├── run.py                # main entry point
├── data.py               # data loading and feature engineering
├── models/               # ARIMA, SARIMA, Dense, CNN, LSTM classes
├── preprocessing/        # notebooks and helpers for preparing data
├── experiments/          # exploratory notebooks and trials
├── results_plots.py      # plotting utilities
├── settings.yaml         # default configuration
└── requirements.txt      # list of dependencies
```

`utils.py` provides helper functions for reading/writing YAML or JSON, and `time_measure.py` implements a simple `TicToc` timer.

## Getting Started

The repository uses Python and only requires a few packages.  You can set up the
environment with `pip`:

```bash
pip install -r requirements.txt
```

The default configuration lives in `settings.yaml`.  It specifies the input
dataset location, aggregation level, model type and other parameters.  You can
run the code with the default settings by executing:

```bash
python run.py
```

Command line options allow you to override the configuration.  Common flags are:

* `--model` – choose between `dense`, `cnn`, `lstm`, `arima` or `sarima`.
* `--online` – enable online (rolling) training.
* `--stations` – list of station names for single‑output mode.
* `--aggregation` – temporal aggregation such as `hour` or `day`.

Example:

```bash
python run.py --model cnn --aggregation day --stations "(1234) example station"
```

Predictions and timing information are saved under the `output` directory.

## Next Steps

Explore the notebooks in `experiments/` and `preprocessing/` to see how the raw data was transformed and how different models were evaluated. Tweaking `settings.yaml` is an easy way to try new parameters or models.
