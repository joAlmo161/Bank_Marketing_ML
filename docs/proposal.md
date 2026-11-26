# Milestone 1 — Project Proposal

**Project title:** Bank Marketing Subscription Prediction Using Machine Learning
**Course:** Introduction to Machine Learning
**Team:** Member A, Member B, Member C, Member D *(replace with real names; roles in `contribution_log.md`)*

## 1. Team members and roles

| Member | Lead role | Leads milestone |
|---|---|---|
| Member A | Project coordinator | 1 (Proposal), 5 (final integration, app) |
| Member B | Data lead | 2 (Data understanding and preprocessing) |
| Member C | Modelling lead | 3 (Baseline and core models), 4 (tuning) |
| Member D | Evaluation and reporting lead | 4 (evaluation, error analysis), 5 (report, presentation) |

## 2. Data set, source and why it is interesting

* **Data set:** Bank Marketing, UCI Machine Learning Repository, **ID 222** (catalogue entry **C2**), file `bank-full.csv`.
* **Creators:** S. Moro, P. Rita and P. Cortez (2014). Licence: CC BY 4.0.
* **Size:** 45,211 rows, 16 input features + target (meets the ≥ 500 rows / ≥ 8 features rule).
* **Why interesting:** it is real data from direct-marketing phone campaigns of a Portuguese bank (2008–2010) and it contains several *real data problems* required by the brief:
  strong class imbalance (≈ 12 % "yes"), categorical features with an `unknown` placeholder, skewed numeric features with outliers, and a **leakage** feature (`duration`) that has to be discussed.

## 3. Problem statement

**What is predicted:** whether a client will subscribe to a term deposit (`y` = yes / no).
**From which inputs:** information available *before* the call — client profile (age, job, marital status, education), financial situation (default, balance, housing and personal loans) and campaign context (contact type, month/day, number of contacts, previous-campaign history).
The call `duration` is excluded because it is only known after the call.
**Who would use it:** the bank's marketing team, to rank clients and call the most promising ones first — fewer wasted calls, less annoyance for uninterested clients, more subscriptions per hour of call-centre time.

## 4. Task type and success measure

* **Task:** binary **classification**.
* **Main success measure:** **ROC-AUC** on 5-fold stratified cross-validation (training data) and on the final test set.
  *Reason:* the classes are imbalanced, so accuracy is misleading (always answering "no" gives ≈ 88 %); ROC-AUC measures how well the model **ranks** subscribers above non-subscribers, which is exactly what a call list needs, and it does not depend on a threshold.
* **Supporting measures:** precision, recall, F1 (positive class) at a chosen threshold, average precision, confusion matrix.
* **Target:** clearly beat the majority-class baseline (ROC-AUC = 0.5) and the logistic-regression model.

## 5. Plan

| Milestone | Content | Lead |
|---|---|---|
| 1 Proposal | This document | Member A |
| 2 Data understanding & preprocessing | Load/verify data, ≥ 5 EDA figures, data problems, split first, `ColumnTransformer` pipeline, imbalance plan | Member B |
| 3 Baseline & core models | Dummy, Logistic Regression, Decision Tree, KNN — same 5-fold stratified CV | Member C |
| 4 Ensembles, tuning & evaluation | Random Forest, Gradient Boosting, GridSearchCV of the best two, single test evaluation, error analysis, bias–variance | Member C / Member D |
| 5 Final | Report, repository, Streamlit deployment (extension), presentation | Member D / all |
