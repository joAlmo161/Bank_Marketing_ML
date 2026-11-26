# Contribution Log

> **Template — please complete with your real team data.**
> The course brief requires "a simple table: date, task, who did it". Replace `Member A–D` with real names,
> adjust dates to your real work sessions, and keep the log consistent with your repository history
> (Git commits). The marker may adjust individual marks based on this log.

## Team and roles

| Member | Lead role (course brief, Section 2) | Main areas |
|---|---|---|
| Member A | Project coordinator | Planning, deadlines, proposal, README, submission, Streamlit app |
| Member B | Data lead | Data loading, EDA (Notebook 01), preprocessing pipeline (Notebook 02) |
| Member C | Modelling lead | Baseline/core models (Notebook 03), ensembles and tuning (Notebook 04) |
| Member D | Evaluation and reporting lead | Final evaluation, error analysis (Notebook 05), figures, report, presentation |

*Teams of 3: merge the roles of Member A and Member D.*
*Roles should rotate between milestones where possible (course brief) — record the rotation below.*

## Log

| Date | Milestone | Task | Who | Hours | Evidence (file / commit) |
|---|---|---|---|---|---|
| YYYY-MM-DD | 1 Proposal | Chose the data set, registered it in the shared file | Member A | | `docs/proposal.md` |
| YYYY-MM-DD | 1 Proposal | Wrote problem statement and success measure | Member A, B | | `docs/proposal.md` |
| YYYY-MM-DD | 2 Data | Download script and data verification | Member B | | `src/data_loader.py` |
| YYYY-MM-DD | 2 Data | EDA figures 1–5 and interpretations | Member B | | `notebooks/01_...ipynb` |
| YYYY-MM-DD | 2 Data | Outlier analysis, preprocessing pipeline, leakage check | Member B, C | | `src/preprocessing.py`, `notebooks/02_...ipynb` |
| YYYY-MM-DD | 3 Models | Baseline and three core models with 5-fold CV | Member C | | `src/train_models.py`, `notebooks/03_...ipynb` |
| YYYY-MM-DD | 3 Models | Validation curve and logistic-regression coefficients | Member C, D | | `notebooks/03_...ipynb` |
| YYYY-MM-DD | 4 Ensembles | Random Forest, Gradient Boosting, GridSearchCV | Member C | | `notebooks/04_...ipynb` |
| YYYY-MM-DD | 4 Ensembles | Threshold selection, learning curve, duration experiment | Member C, D | | `notebooks/04_...ipynb` |
| YYYY-MM-DD | 4 Evaluation | Final test evaluation and error analysis | Member D | | `src/evaluate.py`, `notebooks/05_...ipynb` |
| YYYY-MM-DD | 5 Final | Streamlit application (deployment extension) | Member A | | `app/app.py`, `src/predict.py` |
| YYYY-MM-DD | 5 Final | Final report | All (D leads) | | submitted separately |
| YYYY-MM-DD | 5 Final | Presentation slides and rehearsal | All | | presented in class |
| YYYY-MM-DD | 5 Final | README, requirement mapping, final checks | Member A | | `README.md` |

## Meetings

| Date | Attendees | Decisions | Action items (who, by when) |
|---|---|---|---|
| YYYY-MM-DD | | | |
