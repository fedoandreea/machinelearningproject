# Machine Learning Project 2026: Predicting Pump Flow (Grundfos)

Group project for **Machine Learning, BSc Data Science, ITU, Fall 2026**, in collaboration with **Grundfos**.

We predict the **flow rate Q (`flow_m3_h`)** of a pump from its measurements: speed, power and pressures, plus current, torque, head and temperature in Case 2. Grundfos can then estimate flow without a physical flow meter.

The full brief is in [`docs/ProjectDescription2026_MachineLearning.pdf`](docs/ProjectDescription2026_MachineLearning.pdf).

---

## Hand-in
| | |
|---|---|
| **Deadline** | **18 December, 14:00**, on Wiseflow |
| **What** | One zip with the report (PDF) and the code |
| **Report** | Max 10 pages (front page excluded), font ≥ 11 pt, margins ≥ 2 cm |
| **Code** | Python, descriptive variable names |
| **GenAI** | Any use of GenAI tools must be cited in the report |

## Tasks
| Task | Data | Goal | Notebook |
|---|---|---|---|
| **0** | Case 1 | Linear regression for **power P** from n (krpm) and Q, as the polynomial P = Σ αᵢⱼ nⁱ Qʲ (i, j = 0..3). Use **feature selection** to choose which terms to keep. Report the expression, the test error, and a plot of error vs flow. | `notebooks/task0_power_feature_selection.ipynb` |
| **1** | Case 1 | Linear regression for **Q** from n, P, Δp (and combinations of them). Report the test error and the parameters. | `notebooks/task1_linear_regression.ipynb` |
| **2** | Case 1 | Predict Q with **M1 Decision Tree**, **M2 Feed-forward NN** and **M3 (our choice)**. M1 and M2 each need a **from-scratch** version (NumPy/SciPy only) and a **library** version. Report test errors and hyperparameters, and explain how we chose them. | `notebooks/task2_models_case1.ipynb` |
| **3** | Case 2 | Predict Q with any model **M4** using all features. Train pumps ≠ test pumps, so split by pump (`GroupKFold`). Argue for the metrics, plot flow vs error, discuss hyperparameter tuning. | `notebooks/task3_case2.ipynb` |

The report must cover:
- data cleaning and EDA
- feature extraction and selection
- visualisation
- implementation of M1–M4 (including how we checked correctness)
- comparison and interpretation of results
- an outlook on what Grundfos could change

## Data
| Folder | File | Rows | Notes |
|---|---|---|---|
| `data/case1/` | `train1.csv` / `test1.csv` | 480 / 120 | 8 columns; all rows `stable` |
| `data/case2/` | `train2.csv` / `test2.csv` | 59,950 / 119,900 | 12 columns; 5 training pumps, 10 **different** test pumps; sampled once per second; `stable` and `transient` states |

**Case 1 columns:** `pump_id`, `elapsed_s`, `operating_state`, `speed_rpm`, `power_w`, `inlet_pressure_bar`, `outlet_pressure_bar`, `flow_m3_h` (the target).

**Case 2** adds `measured_head_m`, `current_a`, `torque_nm` and `temperature_c`.

**Rule: never tune anything on the test set.** Choose models and hyperparameters with cross-validation on train, and evaluate on test once, at the end.

## Repository map
```
├── README.md            ← you are here
├── CONTRIBUTING.md      ← HOW WE WORK: branches, pull requests, issues (read this first!)
├── environment.yml      ← conda environment
├── requirements.txt     ← same packages for pip
├── docs/                ← project brief (PDF)
├── data/                ← Case 1 and Case 2 CSVs (do not edit)
├── notebooks/           ← one notebook per task, each with one owner
│   ├── 00_eda.ipynb
│   ├── task0_power_feature_selection.ipynb
│   ├── task1_linear_regression.ipynb
│   ├── task2_models_case1.ipynb
│   └── task3_case2.ipynb
├── src/                 ← shared Python code, imported by the notebooks
│   ├── data.py              loading data + feature engineering
│   ├── decision_tree.py     M1 from scratch
│   └── neural_network.py    M2 from scratch
├── report/              ← report notes + figures/ for exported plots
└── .github/             ← templates for pull requests and issues
```

## Quick start
```bash
git clone https://github.com/fedoandreea/machinelearningproject.git
cd machinelearningproject
conda env create -f environment.yml     # first time only
conda activate mlproject
jupyter lab                             # open a notebook in notebooks/
```
In a notebook, load data through the shared helpers so every notebook builds the features the same way:
```python
import sys; sys.path.append("..")
from src.data import load_case1, add_features
train, test = load_case1()
train, test = add_features(train), add_features(test)
```

## Who does what
Fill this in together and keep it in sync with the GitHub Issues.

| Area | Owner | Status |
|---|---|---|
| EDA + data cleaning (`00_eda`) | | |
| Task 0 – feature selection | | |
| Task 1 – linear regression | Ofri | in progress |
| Task 2 – M1 decision tree (scratch + sklearn) | | |
| Task 2 – M2 neural network (scratch + library) | | |
| Task 2 – M3 | | |
| Task 3 – M4 + PCA (Case 2) | | |
| Report writing + figures | | |
