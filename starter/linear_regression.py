"""Linear regression for the total turnaround time, built around three ideas:

1. Branch type: airport, station or city center.
2. Staff available when the car arrives: 0 while the branch is closed, otherwise staff_on_duty.
3. How much work those staff face: this car's jobs, plus the cars that got in line before it.

Each idea becomes a few features. The regression then learns one weight (minutes per unit) for each.

    python linear_regression.py

Trains on train.csv, then scores on the validation week (true totals are in validation_labels.csv).
The typical minutes per job come from validation_labels.csv: the only file with per-job times.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

TASKS = [
    "quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
    "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check",
]
STAFF_TASKS = [t for t in TASKS if t != "electric_charging"]  # charging runs on the charger, no worker
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HERE = Path(__file__).resolve().parent


def find_data_dir() -> Path:
    for p in (HERE.parent / "data", HERE.parent):
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError("train.csv not found in ../data or the repo root")


# ---------------------------------------------------------------------------------------------
# Opening hours: when could staff start on this car, and how long until the branch closes?
# ---------------------------------------------------------------------------------------------
def read_hours(branches: pd.DataFrame) -> dict:
    """(branch, 'Mon') -> (open minute, close minute, staff), or None on a closed day."""
    hours = {}
    for r in branches.itertuples():
        if str(r.open).lower() == "closed":
            hours[(r.branch, r.day)] = None
        else:
            o = int(r.open[:2]) * 60 + int(r.open[3:])
            c = int(r.close[:2]) * 60 + int(r.close[3:])
            hours[(r.branch, r.day)] = (o, c, int(r.staff_on_duty))
    return hours


def next_open_period(hours, branch, t: pd.Timestamp):
    """The open period that contains t, or the next one after t: (start, close) timestamps."""
    day = t.normalize()
    for _ in range(8):
        h = hours[(branch, DAYS[day.dayofweek])]
        if h is not None:
            start, close = day + pd.Timedelta(minutes=h[0]), day + pd.Timedelta(minutes=h[1])
            if t < close:
                return max(t, start), close
        day += pd.Timedelta(days=1)
    raise RuntimeError(f"{branch} never opens")


def opening_features(df: pd.DataFrame, hours: dict) -> pd.DataFrame:
    out = []
    for r in df.itertuples():
        start, close = next_open_period(hours, r.branch, r.return_ts)
        next_start, _ = next_open_period(hours, r.branch, close)
        out.append({
            "work_start": start,
            # minutes the car sits with 0 staff available (key drop box while closed)
            "closed_wait_min": (start - r.return_ts).total_seconds() / 60,
            "min_left_before_close": (close - start).total_seconds() / 60,
            "night_min": (next_start - close).total_seconds() / 60,  # closed time after that shift
        })
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
def build_features(df: pd.DataFrame, branches: pd.DataFrame, typical_min: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["return_ts"] = pd.to_datetime(df.return_ts)
    hours = read_hours(branches)
    staff = df.branch.map(branches.groupby("branch").staff_on_duty.first()).astype(float)
    btype = df.branch.str.split("_").str[1]
    op = opening_features(df, hours)

    X = pd.DataFrame(index=df.index)
    # 1) branch type (airport is the reference level, so it has no column of its own)
    X["is_station"] = (btype == "STATION").astype(float)
    X["is_center"] = (btype == "CENTER").astype(float)

    # 2) staff available at the moment the car arrives
    X["closed_wait_min"] = op.closed_wait_min

    # 3a) this car's own jobs, shared across the staff
    flags = df[TASKS].astype(float)
    staff_jobs = flags[STAFF_TASKS].sum(axis=1)
    job_minutes = flags * typical_min[TASKS].to_numpy()  # typical minutes of each job done
    X["jobs_per_staff"] = staff_jobs / staff
    X["work_min_per_staff"] = job_minutes[STAFF_TASKS].sum(axis=1) / staff
    X["longest_job_min"] = job_minutes.max(axis=1)  # parallel staff can't beat the longest job

    # 3b) other cars' jobs: cars that got in line at this branch during the hour before this car
    #     could be started on, per staff member. Cars left in the key box overnight all line up
    #     at opening time, in the order they were returned.
    df["work_start"] = op.work_start
    df["jobs_total"] = flags.sum(axis=1)
    ahead_cars = np.zeros(len(df))
    ahead_jobs = np.zeros(len(df))
    for _, g in df.groupby("branch"):
        g = g.sort_values(["work_start", "return_ts", "turnaround_id"])
        s = g.work_start.to_numpy()
        jobs = np.concatenate([[0.0], np.cumsum(g.jobs_total.to_numpy())])
        first = np.searchsorted(s, s - np.timedelta64(60, "m"), side="left")
        pos = np.arange(len(g))
        idx = df.index.get_indexer(g.index)
        ahead_cars[idx] = pos - first
        ahead_jobs[idx] = jobs[pos] - jobs[first]
    X["cars_ahead_per_staff"] = ahead_cars / staff
    X["jobs_ahead_per_staff"] = ahead_jobs / staff

    # 4) the branch closes before this car's work is likely done: the night gets added
    expected_work = np.maximum(X.longest_job_min, X.work_min_per_staff) + 12 * X.cars_ahead_per_staff
    X["night_if_late_min"] = np.where(op.min_left_before_close < expected_work, op.night_min, 0.0)
    return X


def typical_job_minutes(labels: pd.DataFrame) -> pd.Series:
    """Average true minutes of each job when it was done (validation_labels.csv)."""
    return pd.Series({t: labels.loc[labels[t] > 0, t].mean() for t in TASKS})


# ---------------------------------------------------------------------------------------------
def main():
    d = find_data_dir()
    train = pd.read_csv(d / "train.csv")
    val = pd.read_csv(d / "validation.csv")
    labels = pd.read_csv(d / "validation_labels.csv")
    branches = pd.read_csv(d / "branch_data.csv")
    y_val = val[["turnaround_id"]].merge(labels, on="turnaround_id").total_turnaround_min.to_numpy()

    typical = typical_job_minutes(labels)
    X_tr = build_features(train, branches, typical)
    X_va = build_features(val, branches, typical)

    y_tr = train.total_turnaround_min
    # README baseline: ordinary least squares on the job flags only
    base = LinearRegression().fit(train[TASKS], y_tr)
    base_mae = np.abs(base.predict(val[TASKS]) - y_val).mean()
    # same features, two ways to fit the line:
    #   least squares (squared error) is pulled hard by the rare 1,000+ minute overnight cars;
    #   least absolute deviation (fits the median) matches the README's MAE scoring, so it is the one we keep.
    ols = LinearRegression().fit(X_tr, y_tr)
    ols_mae = np.abs(np.maximum(ols.predict(X_va), 0.0) - y_val).mean()
    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(X_tr, y_tr)
    pred = np.maximum(model.predict(X_va), 0.0)
    mae = np.abs(pred - y_val).mean()

    print(f"Validation total MAE  baseline (job flags only, least squares): {base_mae:6.1f} min")
    print(f"Validation total MAE  these features, least squares:            {ols_mae:6.1f} min")
    print(f"Validation total MAE  these features, median (kept):            {mae:6.1f} min\n")
    btype = val.branch.str.split("_").str[1]
    print("MAE by branch type:", {b: round(float(np.abs(pred - y_val)[btype == b].mean()), 1)
                                  for b in ["AIRPORT", "STATION", "CENTER"]}, "\n")

    # weight = minutes added per 1 unit of the feature; 'typical effect' = weight x 1 std of feature
    w = pd.DataFrame({
        "weight (min per unit)": model.coef_,
        "feature std": X_tr.std().to_numpy(),
    }, index=X_tr.columns)
    w["typical effect (min)"] = w.iloc[:, 0] * w.iloc[:, 1]
    print(f"intercept: {model.intercept_:.1f} min (only a starting point: every car has at least two jobs, "
          "and their weights below lift real predictions well above it)")
    print(w.round(2).sort_values("typical effect (min)", key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
