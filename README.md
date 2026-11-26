# Bank Marketing Subscription Prediction Using Machine Learning

*Introduction to Machine Learning — Course Project*
**Team:** Member A (project coordinator) · Member B (data lead) · Member C (modelling lead) · Member D (evaluation and reporting lead)
*(Replace the placeholders with your real names here and in `contribution_log.md`.)*

> **In one sentence:** we predict, *before* a marketing call, whether a bank client will subscribe to a term deposit, using the UCI Bank Marketing dataset. We compare 6 models in a leakage-free scikit-learn pipeline and deploy the best one (tuned Gradient Boosting, test ROC-AUC **0.806**) in a Streamlit app.

---

## Contents
1. [Project idea](#1-project-idea)
2. [Problem statement](#2-problem-statement)
3. [Dataset source and download](#3-dataset-source-and-download-instructions)
4. [Dataset description](#4-dataset-description)
5. [Target variable](#5-target-variable)
6. [Project structure](#6-project-structure)
7. [Installation](#7-installation)
8. [How to run the notebooks](#8-how-to-run-the-notebooks)
9. [How to run the Streamlit interface](#9-how-to-run-the-streamlit-interface)
10. [Models used](#10-models-used)
11. [Evaluation metrics](#11-evaluation-metrics)
12. [Actual final results](#12-actual-final-results)
13. [Limitations](#13-limitations)
14. [References](#14-references)
15. [Educational guide: the project step by step](#15-educational-guide-the-project-step-by-step)
16. [Where is each project requirement implemented?](#16-where-is-each-project-requirement-implemented)
17. [Complete requirement checklist](#17-complete-requirement-checklist)
18. [Verification log and troubleshooting](#18-verification-log-and-troubleshooting)

---

## 1. Project idea
A Portuguese bank contacted clients by phone to sell **term deposits**. Only **11.7 %** of contacted clients subscribed, so most calls were wasted.
We build a **machine learning classification system** that estimates, for each client, the probability of subscribing. The bank can then **call the most promising clients first**.

## 2. Problem statement
* **Task:** binary classification. Predict `y` (yes / no) from information available **before** the call.
* **Users:** the bank's marketing / call-centre team, who would use the ranked list of clients.
* **Success measure:** **ROC-AUC** (main; ranking quality, robust to class imbalance), plus precision, recall and F1 of the "yes" class, and the confusion matrix.
* **Key constraint:** the feature `duration` (length of the current call) is only known after the call → **excluded** (data leakage; see Section 15, step 2).

## 3. Dataset source and download instructions
* **Source:** UCI Machine Learning Repository — *Bank Marketing*, **ID 222**: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
* **Creators:** S. Moro, P. Rita and P. Cortez (2014). DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306). **Licence:** CC BY 4.0 (sharing allowed with attribution).
* **Included:** `data/raw/bank-full.csv` (4.6 MB) and `data/raw/bank-names.txt` are already in this project.
* **Automatic download** (if the file is missing): `python -m src.data_loader`. It downloads the official ZIP and extracts `bank-full.csv` into `data/raw/`. The notebooks also call it automatically.
* **Manual download:** open the UCI page → *Download* → unzip `bank+marketing.zip` → unzip the inner `bank.zip` → copy `bank-full.csv` to `data/raw/`.

## 4. Dataset description
Verified in Notebook 01 (not copied from the documentation):

| Property | Value |
|---|---|
| Rows | **45,211** |
| Columns | **17** = 16 inputs + target `y` |
| Numeric inputs (7) | age, balance, day, duration, campaign, pdays, previous |
| Categorical inputs (9) | job, marital, education, default, housing, loan, contact, month, poutcome |
| Missing values (NaN) | 0 |
| `"unknown"` placeholders | poutcome 36,959 (81.75 %), contact 13,020 (28.80 %), education 1,857 (4.11 %), job 288 (0.64 %) |
| Duplicate rows | 0 |
| Class balance | **5,289 yes (11.70 %)** vs 39,922 no (88.30 %) → imbalance ratio ≈ 7.5 : 1 |
| Special code | `pdays = -1` = never contacted before (81.7 % of rows) |
| Used by the model | 15 inputs (all except `duration`) + 1 engineered flag `previously_contacted` → 51 features after one-hot encoding |

## 5. Target variable
`y` — *has the client subscribed to a term deposit?*
* **yes** (encoded as **1**, the *positive* class): the client subscribed.
* **no** (encoded as **0**): the client did not subscribe.

The app shows **YES** when the predicted probability of "yes" is at least the decision threshold **0.183**, chosen on training data (Section 15, step 7).

## 6. Project structure
```
Bank_Marketing_ML_Project/
├── README.md                    ← this file
├── requirements.txt             ← pinned library versions
├── .gitignore
├── .streamlit/config.toml       ← app colour theme
├── contribution_log.md          ← team log template (date, task, who)
├── data/
│   ├── raw/                     ← bank-full.csv, bank-names.txt (original UCI files)
│   └── processed/               ← train.csv, test.csv (written by Notebook 02)
├── notebooks/
│   ├── 01_Data_Understanding_and_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Baseline_and_Core_Models.ipynb
│   ├── 04_Ensemble_Models_and_Tuning.ipynb
│   └── 05_Final_Evaluation_and_Error_Analysis.ipynb
├── src/                         ← reusable, commented Python code used by notebooks and app
│   ├── config.py                ← paths, random seed, feature lists
│   ├── data_loader.py           ← download, load, data-quality report
│   ├── preprocessing.py         ← split, feature engineering, ColumnTransformer pipeline
│   ├── train_models.py          ← model definitions, CV, GridSearchCV, saving (runnable script)
│   ├── evaluate.py              ← metrics, threshold choice, plots
│   ├── predict.py               ← load model, validate inputs, predict
│   └── plot_style.py            ← colour-blind-friendly figure style
├── app/app.py                   ← Streamlit user interface
├── models/
│   ├── final_model.joblib       ← fitted pipeline (preprocessing + Gradient Boosting)
│   └── model_metadata.json      ← threshold, parameters, valid input values
├── figures/                     ← all 13 figures (PNG) produced by the notebooks (5 EDA + 8 modelling/evaluation)
├── results/                     ← all result tables (CSV / JSON) produced by the notebooks
└── docs/proposal.md             ← Milestone 1 proposal
```

## 7. Installation
Requirements: **Python 3.13** (3.11+ should work), about 1 GB free disk space.

```bash
# 1. Open a terminal in the project folder
cd Bank_Marketing_ML_Project

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate

# 3. Install the pinned libraries
pip install -r requirements.txt
```

The versions are pinned because a saved joblib model is only guaranteed to load with the **same scikit-learn version** (1.9.1). If you must use another version, re-run Notebook 04 to re-create the model.

## 8. How to run the notebooks
Run them **in order** — each one uses files written by the previous one:

| # | Notebook | Produces | Run time* |
|---|---|---|---|
| 1 | `01_Data_Understanding_and_EDA.ipynb` | EDA figures 1–5, `results/data_quality.csv` | ~15 s |
| 2 | `02_Data_Preprocessing.ipynb` | `data/processed/*.csv`, outlier summary table (no figures) | ~10 s |
| 3 | `03_Baseline_and_Core_Models.ipynb` | `results/cv_results_core.csv`, figures 6–7, coefficient table | ~35 s |
| 4 | `04_Ensemble_Models_and_Tuning.ipynb` | tuned model in `models/`, figures 8–9, tuning and learning-curve tables | ~13–18 min (two grid searches: Gradient Boosting and Random Forest) |
| 5 | `05_Final_Evaluation_and_Error_Analysis.ipynb` | test results, figures 10–13, gains and importance tables | ~20 s |

\*Measured on our machine (22 logical CPU cores); slower computers will need longer. All cross-validation uses `n_jobs=-1` (all cores).

Options:
* **VS Code:** open a notebook → select the kernel of the Python where you installed `requirements.txt` (top-right *Select Kernel*; e.g. *Python 3.13.9* or your `.venv`) → **Restart** → **Run All**. Always run the cells **from the top**: later cells need the imports and variables of earlier cells.
* **Jupyter:** `pip install notebook` then `jupyter notebook notebooks/` (on Windows, enable long paths if installation fails, see Section 18).
* **Command line (headless):** `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=6000 notebooks/01_Data_Understanding_and_EDA.ipynb` (repeat for 02–05).
* **Google Colab:** upload the whole folder (or zip) to Google Drive, open a notebook, run `%cd /content/drive/MyDrive/Bank_Marketing_ML_Project` and `!pip install -r requirements.txt` first.
* **Script only (no notebooks):** `python -m src.train_models` repeats the full training (CV of all models, tuning, threshold, saving the model) in one command.

All outputs are already stored in the notebooks, so they can be **read without running them**.

## 9. How to run the Streamlit interface
From the project root (with the environment activated):
```bash
streamlit run app/app.py
```
Your browser opens at <http://localhost:8501>. The app has three tabs:
1. **Predict a client:** fill in the form (age, job, marital status, education, credit default, balance, housing loan, personal loan, contact type, month/day, number of contacts, previous-campaign information) → press **Predict subscription**. You get **YES** or **NO**, the probability, and warnings for missing or unusual inputs.
2. **About the project:** problem explanation and data source.
3. **Model performance:** the real test metrics and figures.

The app loads the **real saved pipeline** (`models/final_model.joblib`); nothing is random or hard-coded. Invalid inputs (e.g. an unknown category) show an error message instead of crashing. Empty fields are allowed and imputed.

## 10. Models used
| Model | Role | Key settings |
|---|---|---|
| DummyClassifier (most frequent) | baseline | always predicts "no" |
| Logistic Regression | simple linear model | `class_weight="balanced"`, `max_iter=2000` |
| Decision Tree | core algorithm | `class_weight="balanced"` (default depth → shows over-fitting) |
| K-Nearest Neighbors | core algorithm | `k=5` |
| Random Forest | ensemble (bagging) | `class_weight="balanced"` |
| **Gradient Boosting** | ensemble (boosting) — **final model** | tuned: `learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.8` |

All models use the same preprocessing pipeline, the same 5 stratified folds and the same metrics. `random_state=42` everywhere.

## 11. Evaluation metrics
* **ROC-AUC (main metric):** the probability that a random subscriber is ranked above a random non-subscriber. It does not depend on the class ratio or on a threshold, and it measures ranking, which is what a call list needs.
* **Average precision (PR-AUC):** focuses on the rare "yes" class (random = 0.117).
* **Precision / Recall / F1** of the "yes" class at the chosen threshold, and the **confusion matrix**.
* **Accuracy** is reported but **not** used for decisions: always answering "no" already gives 88.3 %.

## 12. Actual final results
All numbers below were produced by the notebooks (`results/*.csv|json`); none are typed by hand or estimated.

**Cross-validation on the training set (5-fold stratified, mean ± std, default settings):**

| Model | ROC-AUC | Average precision | Train ROC-AUC |
|---|---|---|---|
| Dummy (majority class) | 0.500 ± 0.000 | 0.117 | 0.500 |
| Logistic Regression | 0.764 ± 0.003 | 0.398 | 0.769 |
| Decision Tree | 0.605 ± 0.006 | 0.172 | 1.000 |
| K-Nearest Neighbors | 0.692 ± 0.008 | 0.283 | 0.919 |
| Random Forest | 0.778 ± 0.005 | 0.404 | 1.000 |
| Gradient Boosting | 0.792 ± 0.004 | 0.439 | 0.807 |
| Random Forest (tuned) | 0.7947 ± 0.0035 | 0.442 | 0.940 |
| **Gradient Boosting (tuned) — final** | **0.7995 ± 0.0025** | **0.449** | 0.862 |

**Single final evaluation on the untouched test set (9,043 clients, threshold 0.183):**

| Metric | Final model | Baseline (majority) |
|---|---|---|
| ROC-AUC | **0.806** | 0.500 |
| Average precision | **0.469** | 0.117 |
| Recall (subscribers found) | **0.540** | 0.000 |
| Precision (flagged clients who subscribe) | **0.467** | 0.000 |
| F1-score | **0.500** | 0.000 |
| Accuracy | 0.874 | 0.883 |
| Confusion matrix | TN 7,332 · FP 653 · FN 487 · TP 571 | — |

* Calling the **top 10 % / 20 %** of clients ranked by the model reaches **45.4 % / 62.7 %** of all subscribers (random: 10 % / 20 %).
* Over-fitting check: ROC-AUC train 0.852 · CV 0.800 · test 0.806 → mild over-fitting; CV and test agree.
* Leakage demonstration: adding `duration` would raise CV ROC-AUC to 0.935. This model is not usable in practice.

## 13. Limitations
* Data from **one Portuguese bank, 2008–2010**; not valid elsewhere or today without re-training.
* **Temporal drift:** the yes-rate is 4.8 % in the first half of the (time-ordered) file vs 18.6 % in the second; a random split may be optimistic for future campaigns.
* **Modest precision** (0.47): the model prioritises calls; it does not guarantee outcomes.
* Most missed subscribers have **no previous campaign history** or an **unknown contact type**.
* **Ethics:** age/job-based targeting raises fairness concerns; personal financial data require GDPR compliance; the model must not be used for credit or pricing decisions, nor to justify repeated calls.

## 14. References
* Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
* Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
* Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
* Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
* Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232.
* Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD, 6*(4), 1–21.
* Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.
* Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2nd ed.). Springer.
* Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
* Library documentation: pandas, NumPy, matplotlib, seaborn, Streamlit.

---

## 15. Educational guide: the project step by step
Read this section before the oral discussion. Each step names the file where the code lives.

**Step 1 — Load and verify the data** (`src/data_loader.py`, Notebook 01).
`load_raw_data()` reads the CSV (separator `;`). We *assert* the shape is 45,211 × 17, so a wrong file fails loudly. `data_quality_report()` counts NaN **and** the word `"unknown"`, the dataset's hidden missing value.

**Step 2 — Explore (EDA) and spot problems** (Notebook 01).
We look at the subscription *rate* per group (not raw counts, because of the imbalance). Five figures: the class imbalance (Figure 1), a U-shaped age effect (Figure 2), a strong effect of the previous outcome (64.7 % after a success, Figure 3), a strong month effect (Figure 4) and weak linear correlations (Figure 5). Section 6 explains the **duration leakage**: the call length is only known after the call, so it cannot be used to decide whom to call.

**Step 3 — Split first** (`split_data` in `src/preprocessing.py`).
80 % train / 20 % test, `stratify=y` (same 11.7 % yes in both) and `random_state=42`. The test set is then locked away until Step 8. *Why first?* Anything learned from the data (means, categories) must come from training data only; otherwise test information leaks into the model.

**Step 4 — Preprocessing pipeline** (`build_preprocessor`).
* `FunctionTransformer` adds `previously_contacted` (pdays = −1 means never contacted).
* Numeric: `SimpleImputer(median)` → `StandardScaler` (mean 0, std 1; needed by KNN).
* Categorical: `SimpleImputer("unknown")` → `OneHotEncoder(handle_unknown="ignore")` (one 0/1 column per category; no fake order).
* Combined with `ColumnTransformer`, then put in a `Pipeline` **with the model**, so `fit` learns everything from the training data only — also inside every CV fold.
* Outliers are analysed (IQR) but kept: they are real clients.

**Step 5 — Handle the imbalance.**
Stratified splits/folds, `class_weight="balanced"` (a "yes" counts 7.55× more in the loss), metrics that focus on the "yes" class, and a tuned threshold. No synthetic oversampling.

**Step 6 — Compare models fairly** (`cross_validate_models`, Notebooks 03–04).
`StratifiedKFold(5)` on the training set: train on 4 folds, validate on 1, repeat 5×, and report mean ± std. `return_train_score=True` gives training scores to detect over-fitting (train ≫ validation) or under-fitting (both low).

**Step 7 — Tune, select, choose threshold** (`tune_model`, `find_best_threshold`, Notebook 04).
`GridSearchCV` tries every combination in `PARAM_GRIDS` with the same folds and picks the best mean ROC-AUC (training data only). The best tuned model wins. We then choose the probability cut-off that maximises F1 on **out-of-fold** training predictions (`cross_val_predict`) → 0.183. The fitted pipeline is saved with `joblib`.

**Step 8 — Final test, once** (Notebook 05).
The saved model predicts the test set one time: ROC-AUC 0.806, recall 0.540, precision 0.467. Then: confusion matrix, ROC and PR curves, a cumulative-gains table, train-vs-CV-vs-test comparison (bias–variance), permutation importance, and error analysis by customer group.

**Step 9 — Deploy** (`src/predict.py`, `app/app.py`).
The app loads the saved pipeline, validates the form inputs, calls `predict_proba`, compares with 0.183 and shows YES/NO with the probability.

**Key vocabulary:** *data leakage* (information not available at prediction time, or test information, used in training) · *over-fitting* (memorising the training data: high variance) · *under-fitting* (too simple: high bias) · *cross-validation* (repeated train/validate splits of the training data) · *stratification* (keeping the class ratio in every split) · *threshold* (probability cut-off for "yes").

---

## 16. Where is each project requirement implemented?

**How to read the locations.** *Line ranges* refer to the final `.py` files in this ZIP and were computed automatically from the source code (Python `ast` module / text search) after the files were finalised — they were not estimated. *Notebook cells* are counted from the top of the notebook starting at 1, **including markdown cells** (in VS Code or Jupyter, count every cell). The section heading is given as well, so you can also search for it.

### 16.1 Data, proposal and EDA (Milestones 1–2)
| Course requirement | File | Location | Explanation |
|---|---|---|---|
| Proposal: team, roles, data set, problem, task type, success measure, plan | `docs/proposal.md` | Sections 1–5 | Milestone 1 document |
| Public data set from the catalogue (C2, UCI ID 222) | `src/data_loader.py` | `download_dataset` lines 23–43 | Downloads official UCI ZIP; URL in `src/config.py` line 34 |
| Load the data | `src/data_loader.py` | `load_raw_data` lines 49–58 | Reads `bank-full.csv` with `sep=";"` |
| Check shape against documented size | `01_Data_Understanding_and_EDA.ipynb` | Cells 4–5 ("2. Data Loading") | `assert df.shape == (45211, 17)` |
| Description of data (rows, columns, types, target) | `01_Data_Understanding_and_EDA.ipynb` | Cells 6–7 ("3. Column Description") | Table of all 17 columns + dtypes |
| Missing values | `src/data_loader.py`; Notebook 01 | `data_quality_report` lines 73–93; NB01 Cells 8–10 ("4.1 Missing Values…") | Counts NaN and `"unknown"` per column |
| Duplicate records | `01_Data_Understanding_and_EDA.ipynb` | Cells 11–13 ("4.2 Duplicate Records") | `df.duplicated().sum()` = 0 |
| Target distribution / class balance | `01_Data_Understanding_and_EDA.ipynb` | Cells 19–21 ("Figure 1") | Bar chart + imbalance ratio; `summarize_dataset` lines 96–108 |
| ≥ 5 meaningful EDA figures/tables, each interpreted | `01_Data_Understanding_and_EDA.ipynb` | Cells 17–33 (Figures 1–5) | 5 figures, each followed by a numbered caption and a short interpretation |
| Correlations | `01_Data_Understanding_and_EDA.ipynb` | Cells 31–33 ("Figure 5") | Heatmap of numeric features + target |
| Relation of key features with target | `01_Data_Understanding_and_EDA.ipynb` | Cells 22–30, Cells 34–36 | Rates by age group, previous-campaign outcome and month (Figures 2–4); duration leakage (Section 6) |
| Data problems found and how handled (summary) | `01_Data_Understanding_and_EDA.ipynb` | Cell 37 ("7. Summary of EDA Findings") | Table: finding → consequence |

### 16.2 Preprocessing (Milestone 2)
| Course requirement | File | Location | Explanation |
|---|---|---|---|
| Train/test split **before** any fitting, stratified | `src/preprocessing.py`; Notebook 02 | `split_data` lines 34–53; NB02 Cells 4–7 | `train_test_split(..., stratify=y, random_state=42)`; saves `data/processed/` |
| Fixed `random_state` everywhere | `src/config.py` | `RANDOM_STATE` line 41 | Imported by split, CV, all models, learning curve, permutation importance |
| Leakage feature removed (`duration`) | `src/config.py`; `src/preprocessing.py` | `LEAKAGE_FEATURE` line 49; `split_data` lines 34–53 | Only added when `include_duration=True` (secondary experiment) |
| Outliers | `src/preprocessing.py`; Notebook 02 | `iqr_outlier_table` lines 147–165; NB02 Cells 8–10 | IQR outlier summary table, decision to keep |
| Inconsistent categories / "unknown" | `02_Data_Preprocessing.ipynb` | Cells 11–13 | Lists all categories; "unknown" kept |
| Skewed features | Notebooks 01–02 | NB01 Cells 14–16; NB02 outlier table (skewness column) | Discussed; handled by scaling + tree models |
| Feature engineering / selection | `src/preprocessing.py`; Notebook 02 | `add_engineered_features` lines 59–73; NB02 Cells 14–16 | `previously_contacted` flag; justification for dropping only `duration` |
| Encoding (OneHotEncoder) and scaling (StandardScaler) with reasons | `src/preprocessing.py` | `build_preprocessor` lines 84–129 | Numeric: SimpleImputer + StandardScaler; categorical: SimpleImputer + OneHotEncoder |
| `ColumnTransformer` + `Pipeline` that avoids leakage | `src/preprocessing.py`; Notebook 02 | `build_preprocessor` lines 84–129, `build_model_pipeline` lines 132–141; NB02 Cells 17–23 | Fit on train only; proof that scaler means = train means |
| SimpleImputer | `src/preprocessing.py` | inside `build_preprocessor` lines 84–129 | median (numeric), constant "unknown" (categorical) |
| Imbalance plan (class weights / training-only resampling / metrics) | `02_Data_Preprocessing.ipynb`; `src/train_models.py` | NB02 Cells 24–25; `get_baseline_and_core_models` lines 63–89 | Stratification, `class_weight="balanced"`, suitable metrics, threshold |

### 16.3 Baseline and core models (Milestone 3)
| Course requirement | File | Location | Explanation |
|---|---|---|---|
| Baseline model (DummyClassifier) | `src/train_models.py`; Notebook 03 | `get_baseline_and_core_models` lines 63–89; NB03 Cells 9–11 | Majority class; 88.3 % accuracy, 0 recall |
| Simple linear model (Logistic Regression) | same | same function; NB03 Cells 12–15 | Balanced class weights |
| ≥ 3 algorithms (LogReg, Decision Tree, KNN) | same | same function; NB03 Cells 12–15 | 3 core algorithms |
| One sentence on suitability + one on weakness per model | `03_Baseline_and_Core_Models.ipynb`; `04_…ipynb` | NB03 Cell 7 ("3. The Models" table); NB04 Cell 4 | Tables "Why suitable / Main weakness" |
| Same metrics, same validation scheme, results table | `src/train_models.py`; Notebook 03 | `CV` line 46, `SCORING` lines 50–57, `cross_validate_models` lines 140–155; NB03 Cells 12–15 | Mean ± std table + Figure 6 |
| Short explanation of results | `03_Baseline_and_Core_Models.ipynb` | interpretation cells after each table; summary Cell 25 | |

### 16.4 Ensembles, tuning and evaluation (Milestone 4)
| Course requirement | File | Location | Explanation |
|---|---|---|---|
| ≥ 2 ensembles (Random Forest, Gradient Boosting) | `src/train_models.py`; Notebook 04 | `get_ensemble_models` lines 92–105; NB04 Cells 4–5 | Bagging and boosting |
| Compared with earlier models | `04_Ensemble_Models_and_Tuning.ipynb` | Cells 6–10 | 6-model table + Figure 8 |
| Stratified cross-validation on training data, mean and spread | `src/train_models.py` | `CV` line 46, `cross_validate_models` lines 140–155; `summarize_cv` in `src/evaluate.py` lines 151–165 | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| Hyperparameter tuning (grid search) of best 1–2 models, training data only | `src/train_models.py`; Notebook 04 | `PARAM_GRIDS` lines 110–134, `tune_model` lines 161–180; NB04 Cells 11–17 | `GridSearchCV`, `refit="roc_auc"`; default-vs-tuned table |
| Metrics that fit the problem + justification of main metric | `src/evaluate.py`; Notebook 03 | module docstring lines 2–22, `classification_metrics` lines 51–60; NB03 Cell 6 | Precision, recall, F1, ROC-AUC, PR-AUC, accuracy |
| Model selection + threshold on training data only | `src/train_models.py`; `src/evaluate.py`; Notebook 04 | `out_of_fold_scores` lines 199–204, `find_best_threshold` lines 66–79; NB04 Cells 18–22 | Rule fixed before testing; Figure 9 |
| Save the model (joblib) | `src/train_models.py`; Notebook 04 | `save_final_model` lines 207–232; NB04 Cells 29–30 | `models/final_model.joblib` + metadata JSON |
| Single final evaluation on the untouched test set | `05_Final_Evaluation_and_Error_Analysis.ipynb` | Cells 4–7 | Test scored once; baseline comparison |
| Confusion matrix | `src/evaluate.py`; Notebook 05 | `plot_confusion_matrix` lines 92–111; NB05 Cells 8–10 | Figure 10 |
| ROC curve | `src/evaluate.py`; Notebook 05 | `plot_roc_curve` lines 114–126; NB05 Cells 11–13 | Figure 11 |
| Precision-Recall curve | `src/evaluate.py`; Notebook 05 | `plot_precision_recall_curve` lines 129–148; same cells | Figure 12 |
| Training vs validation performance | Notebooks 03–05 | NB03 Cells 16–18; NB04 Cells 6–10; NB05 Cells 17–19 | Train / CV / test tables |
| Over-fitting / under-fitting discussion | Notebooks 03–04 | NB03 Cells 19–21 (validation curve, Figure 7) | Depth 1–3 under-fit, > 8 over-fit |
| Bias–variance trade-off | Notebooks 04–05 | NB04 Cells 23–25 (learning-curve table); NB05 Cells 17–19 | |
| Error analysis (groups, examples, reasons, improvements) | `05_Final_Evaluation_and_Error_Analysis.ipynb` | Cells 23–30 | Recall/precision per group (Figure 13), FN/FP examples and profiles, ideas |
| Duration with/without comparison | `04_Ensemble_Models_and_Tuning.ipynb` | Cells 26–28 | CV only, for discussion |
| Feature importance (interpretation) | `05_Final_Evaluation_and_Error_Analysis.ipynb` | Cells 20–22 | Permutation importance (table of the 8 most important features) |

### 16.5 Extension, deployment, report, repository (Milestone 5)
| Course requirement | File | Location | Explanation |
|---|---|---|---|
| Extension: deployment | `app/app.py` | whole file; model loading lines 54–57 | Streamlit app |
| Customer input form | `app/app.py` | lines 110–169 | 3-column form, previous-campaign toggle |
| Prediction button + result + probability | `app/app.py` | lines 171–206 | YES/NO card, probability, threshold |
| Input validation / missing inputs | `src/predict.py` | `validate_customer` lines 37–89 | Errors vs warnings; imputation of missing values |
| Prediction with the real model | `src/predict.py` | `load_model` lines 26–34, `predict_customer` lines 92–111 | joblib + `predict_proba` + saved threshold |
| Project introduction / problem explanation in app | `app/app.py` | lines 211–230 | "About the project" tab |
| Full training as a script | `src/train_models.py` | `main` lines 238–268 | `python -m src.train_models` |
| Final report (PDF, 8–12 pages, captions, references) | submitted separately (not in this repository) | — | 14 sections, APA references |
| README (title, team, problem, data, install/run, structure, results, reproduce) | `README.md` | this file | |
| requirements.txt | `requirements.txt` | — | Pinned versions |
| Data file or download instructions | `data/raw/`; `src/data_loader.py` | `download_dataset` lines 23–43 | Both included |
| Contribution log (date, task, who) | `contribution_log.md` | — | Template to complete |
| Presentation (10–12 min, every member) + Q&A | presented in class (not in this repository) | — | 14 slides, 4 presenters, 16 prepared answers |
| Notebook runs top to bottom without errors | all notebooks | — | Verified (Section 18) |

## 17. Complete requirement checklist
Mapped to the course brief's **Technical Requirements (Minimum Standard)** and milestone contents.

**Technical requirements (Section 6 of the brief)**
- [x] Fix `random_state` everywhere → `src/config.py` `RANDOM_STATE = 42`, used in split, CV, models, searches, curves.
- [x] Split the data first; fit all preprocessing inside a Pipeline → `split_data`, `build_model_pipeline`; proof in NB02 Section 6.1.
- [x] Include a baseline model → `DummyClassifier`, NB03 Section 4.
- [x] At least 3 algorithms plus 2 ensembles → LogReg, Decision Tree, KNN + Random Forest, Gradient Boosting (exactly six models including the baseline).
- [x] Cross-validation and tuning on training data only → `StratifiedKFold`, `GridSearchCV` (NB03–04 never use `X_test`).
- [x] Use the test set once, at the end → NB05 Section 2.
- [x] Choose metrics that match the problem → ROC-AUC (main), PR-AUC, F1, precision, recall; accuracy shown to be misleading.
- [x] Report error analysis and limitations → NB05 Section 8; README Section 13.
- [x] Notebooks run from top to bottom without errors → Section 18.

**Milestone 1 — Proposal** → `docs/proposal.md`
- [x] Team members and roles · [x] Data set, source, why interesting · [x] Problem statement (what, from which inputs, who uses it) · [x] Task type and success measure with reason · [x] Plan per milestone with leads.

**Milestone 2 — Data understanding and preprocessing**
- [x] Rows, columns, types, target, class balance (NB01 §2–4, Figure 1)
- [x] ≥ 5 meaningful figures/tables, each with interpretation (NB01: 5 figures)
- [x] Missing values, duplicates, outliers, inconsistent categories, skewed features (NB01 §4, NB02 §3–4)
- [x] Encoding and scaling choices with reasons; feature engineering/selection (NB02 §5–6)
- [x] Train/test split before fitting; ColumnTransformer pipeline without leakage (NB02 §2, §6)
- [x] Justified imbalance plan (NB02 §7)

**Milestone 3 — Baseline and core models**
- [x] Baseline (majority class) and simple linear model (logistic regression)
- [x] ≥ 3 algorithms from Units 4–5
- [x] Results table, same metrics and validation scheme, short explanation
- [x] One sentence on suitability and one on weakness per model

**Milestone 4 — Ensembles, tuning and evaluation**
- [x] ≥ 2 ensembles compared with earlier models
- [x] Stratified CV on training data with mean and spread
- [x] Grid search of the best two models on training data only
- [x] Suitable metrics and justification of main metric
- [x] Single final test evaluation with confusion matrix (+ ROC and PR curves)
- [x] Error analysis: failing groups/examples, reasons, improvement ideas
- [x] Over-/under-fitting (train vs validation vs test) and bias–variance

**Milestone 5 — Final report, code, presentation**
- [x] Report PDF with the required sections, figures with captions discussed in the text, APA references (submitted separately)
- [x] Repository/ZIP with notebooks, scripts, data + download script, `requirements.txt`
- [x] README with title, team, problem, data source, install/run, structure, main results, reproduction
- [x] Readable code: clear names, comments that explain *why*, no unused experiments, outputs visible in notebooks
- [x] Presentation outline (10–12 min, every member presents) + prepared answers (presented in class)
- [x] Contribution log template
- [x] Extension: deployment (Streamlit app)
- [ ] **To do by the team:** fill in real names, dates and contributions; commit regularly to a shared Git repository; rehearse the talk.

## 18. Verification log and troubleshooting
Checks actually executed before delivery (Windows 11, Python 3.13.9, library versions as in `requirements.txt`):

| Check | How | Result |
|---|---|---|
| Dataset download | `download_dataset(force=True)` into a temporary folder | file identical (MD5) to `data/raw/bank-full.csv` |
| Dataset statistics | Notebook 01 `assert` + outputs | 45,211 × 17, 5,289 yes (11.70 %), 0 NaN, 0 duplicates |
| All 5 notebooks, clean run | deleted `figures/`, `results/`, `models/`, `data/processed/`, then `nbconvert --execute` on 01→05 | all exited with code 0, no error cells |
| Kernel used for the final test | global **Python 3.13.9**, Jupyter kernel `python3` (the kernel VS Code offers) with the pinned versions of `requirements.txt` | all five notebooks executed 01→05 from a fresh kernel: exit code 0, every code cell executed in order, 0 error outputs, 0 stderr outputs |
| Out-of-order execution | ran only the learning-curve cell of Notebook 04 in a fresh kernel | before the fix: `NameError: name 'learning_curve' is not defined`; after: a clear `RuntimeError` asking to run from the top |
| Results preserved after the fix | compared with a backup made before re-running | result tables, threshold (0.1828), model metadata and the test probabilities of all 9,043 clients identical |
| EDA reduced to 5 figures, notebooks restructured, other charts reduced to 8 (outlier box plots, LR coefficients, learning curve, gains and permutation-importance charts replaced by tables), figures renumbered 1–13 | `nbconvert --execute` on 01→05; result files compared with a backup | 0 error cells; all 13 figures regenerated; all result tables, model metadata and test probabilities (max. difference 0.0) identical, except the stored Decision Tree CV row, which now matches the documented values (0.605 ± 0.006) — the earlier stored output (0.606 ± 0.009) came from an older run |
| Model loading | `python -m src.predict` | prediction returned from the saved joblib pipeline |
| Training script | `python -m src.train_models` (15 min) | same model choice, parameters and threshold; test probabilities identical to the notebook model (max. difference 0.0) |
| Streamlit app | `streamlit.testing` (AppTest): default inputs, a "previous success" client, missing fields; plus `streamlit run app/app.py` headless | no exceptions; real probabilities (e.g. 84.8 % → YES); warnings for missing inputs; server health check `ok`, HTTP 200 |
| Invalid input | `predict_customer({... "age": "abc", "job": "astronaut"})` | clear error messages, no crash |
| PDF report (submitted separately) | built with the project's report-builder script; page count with `pypdf` | 11 pages: title page + 9 pages of body text (Sections 1–13) + 1 page of references |
| README line numbers | computed with `ast` / text search from the final files; spot-checked by printing those lines | all references resolved |

**Not verified by us:** the app was tested programmatically and via its HTTP endpoint, but no browser screenshot was taken. Open it yourself with `streamlit run app/app.py`.

**Troubleshooting**
* *`NameError: name '…' is not defined` (e.g. `learning_curve`, `final_model`)* → a cell was run before the cells above it (typically after a kernel restart). Click **Restart**, then **Run All**. In Notebook 04 the cells after the grid search stop with a clear message telling you which section to run first.
* *`ModuleNotFoundError: No module named 'sklearn'` (or pandas, seaborn…)* → VS Code is using a Python without the libraries. On this computer `py` defaults to **Python 3.14**, which does not have them; select the **Python 3.13.9** kernel (or install `requirements.txt` into the interpreter you select).
* *`ModuleNotFoundError: No module named 'src'`* → run commands from the project root, and start notebooks from the `notebooks/` folder (the first cell adds the project root to the path).
* *`pip install notebook` / `jupyterlab` fails on Windows with a long-path error* → enable long paths (`Settings → System → For developers → Enable long paths`, or registry `LongPathsEnabled = 1`), or use a shorter folder such as `C:\ml_project`. VS Code needs only `ipykernel`.
* *Warnings like `KeyError: ...joblib_memmapping_folder...` at the end of a run* → harmless clean-up messages from joblib's parallel workers on Windows; results are unaffected.
* *`InconsistentVersionWarning` or errors when loading the model* → install the pinned `scikit-learn==1.9.1` or re-run Notebook 04.
* *Slow Notebook 04* → the two grid searches (180 fits each) take about 18 min on our machine; results are already stored in the notebook outputs, so you only need to re-run it to reproduce them.
