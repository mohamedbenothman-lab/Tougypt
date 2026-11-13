# Task Time Detective: how long does each turnaround task take?

Project owner: Iulia Pasov (iulia.pasov@sixt.com). Questions before the hackathon: email the owner. On the day, ask on site.

## Problem
When a customer returns a vehicle we prepare it for the next customer. Every turnaround includes an exterior cleaning
(quick or normal) and an interior cleaning, and may include special cleaning, damage check, refuel, electric charging,
quick damage repair and tyre check. We know which tasks were performed and the **total** time until the car was ready, but
**not how long each task took**. Knowing task times lets us plan staff and tell customers when a car will be ready.

## Your mission
Estimate how long each task takes, and predict the total turnaround time for new turnarounds.
1. Explore what drives the total, and what drives each task.
2. Build a model that outputs a duration for every task performed (your estimate of its time in minutes) and a total.
3. Explain it: which features matter for which task, and does it make sense?

`total_turnaround_min` is the time from `return_ts` until the car was ready. Real operations are messier than a sum of task
times, so look at the data closely.

## Data (`data/`, see `data/DATA_DESCRIPTION.md`)
- `train.csv`: about 34,000 turnarounds with total time, no per-task times.
- `validation.csv` + `validation_labels.csv`: one later full week, Monday to Sunday (about 2,800 turnarounds) **with the true total and the
  true duration of each task**, so you can check your estimates. Use it however you like.
  Every turnaround has interior cleaning and exactly one exterior cleaning, so the totals alone barely separate the two.
- `branch_data.csv`: branch opening hours and staff.
We run your code on a **hidden test set** from the three weeks after the validation week.

## How results are judged (on the hidden test set)
1. **Per-task duration error (main).** Your estimate for each task performed, compared to the true duration: MAE per task, the
   mean across tasks (every task counts equally, rare ones too), and the weighted error (WAPE: total absolute error divided by
   total true task minutes). `evaluate.py` also prints a skill score per task against always guessing the task's average duration
   in `validation_labels.csv`; it is harsh for tasks whose duration barely varies, so we use it for information only.
2. **Total turnaround error.** MAE of `total_turnaround_min`.
3. **Plausibility and feature importance.** No negative or impossible durations; effects a branch manager would believe.
4. **Reproducibility and clarity.** Our run must give the same values as yours; someone else should be able to follow your approach.

## What good looks like (total MAE and per-task WAPE, on validation and on the hidden test)
- **Baseline** (`solution.py` as given): total MAE about 99 min, WAPE about 0.55. It is a floor, not a target: it is worse than
  guessing each task's average duration.
- **Good:** total MAE under 15 min.
- **Strong:** total MAE under 5 min and WAPE under 0.2.
Reproducibility and plausibility are required for any of these to count.

## What you submit
- `solution.py` in `starter/`, with `fit(train, branches)` and `predict(model, df, branches)` (see the template).
- `SUBMISSION.md`: approach, results on validation, feature importance, how to run, per-task estimate table.
- Either a 1-minute demo (`submission/demo.mp4`, under 50 MB) or up to 5 slides (`submission/slides.pdf`).
Put them in your project folder in the repo. Dependencies go in `requirements.txt` (in this folder), with pinned versions (`package==x.y.z`).
Your predictions must include a `dur_<task>` column for every task: a total without task estimates is not accepted.

## Rules
- Your code must be **deterministic**: fix random seeds; two runs must give identical values. We re-run it and compare.
- Must run in under 10 minutes on a laptop, no network, nothing outside `starter/` and `data/` except `requirements.txt`.
- No hard-coded labels or durations that you did not learn from the data files.
- The hidden test set is not in this repo. Do not try to obtain it.

## Setup (before the day)
`git`, Python 3.11+, a virtual environment, `bash setup.sh` (or `pip install -r requirements.txt`), your own LLM coding agent (none is
provided). Never commit secrets. Check it works:
```
cd starter
python run.py --repeat            # baseline: writes predictions.csv for validation.csv and checks two runs are identical
python evaluate.py predictions.csv ../data/validation_labels.csv --task-truth ../data/validation_labels.csv --flags ../data/validation.csv \
    --train-truth ../data/validation_labels.csv
```
The baseline (linear regression of the total on the task flags) is the bar to beat.
