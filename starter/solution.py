"""Task Time Detective: per-task durations and total turnaround time.

How the data works (found by checking the validation labels; see SUBMISSION.md):

* Every task has its own duration. Per-task models learn it from the labelled validation week
  (gradient boosting, absolute-error loss).
* The staff tasks of one car are split across the branch's `staff_on_duty` workers, longest task
  first (LPT list scheduling). Electric charging runs in parallel on the charger.
  makespan = max(LPT(staff tasks, staff_on_duty), charging)
* Before work starts, a car waits for the cars already queued at the branch, roughly 12 min / staff
  per car ahead. Service times in that queue are random, so the wait is estimated from the branch's
  recent arrivals (a model trained on the `train.csv` totals).
* Work only happens while the branch is open. A car returned to the key box, or work running past
  closing, continues at the next opening.
  ready = return + (wait + makespan) counted in *open* minutes.

Interface: fit(train, branches) -> model, predict(model, df, branches) -> DataFrame.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 0
TASKS = [
    "quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
    "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check",
]
STAFF_TASKS = [t for t in TASKS if t != "electric_charging"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CAT_LEVELS = {
    "vehicle_class": ["compact", "midsize", "premium", "suv", "van"],
    "powertrain": ["petrol_diesel", "hybrid", "bev"],
    "weather": ["dry", "rain", "snow"],
}
NUM_COLS = [
    "rental_days", "km_driven", "return_hour", "weekday", "fuel_level_pct", "battery_soc_pct",
    "interior_dirt", "exterior_dirt", "queue_length",
]
FLAG_COLS = ["child_seat", "pet", "smoking_reported", "damage_reported"]


# --------------------------------------------------------------------------------------------
# Opening hours: an "open-minute clock" per branch (minutes the branch has been open since a
# reference day). Work and waiting only advance this clock.
# --------------------------------------------------------------------------------------------
def _hhmm(s) -> float:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


class OpenClock:
    def __init__(self, branches: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        self.ref = start.normalize() - pd.Timedelta(days=1)
        n_days = int((end.normalize() - self.ref).days) + 40  # room for closures after the last return
        self.staff = branches.groupby("branch").staff_on_duty.first().astype(int).to_dict()
        self.tables = {}
        dows = (self.ref.dayofweek + np.arange(n_days)) % 7
        for b, g in branches.groupby("branch"):
            hours = {}
            for _, r in g.iterrows():
                closed = str(r.open).lower() == "closed" or pd.isna(r.close) or str(r.close) == ""
                hours[r.day] = (0.0, 0.0) if closed else (_hhmm(r.open), _hhmm(r.close))
            o = np.array([hours[DAYS[d]][0] for d in dows])
            c = np.array([hours[DAYS[d]][1] for d in dows])
            length = c - o
            cum = np.concatenate([[0.0], np.cumsum(length)])  # open minutes before each day
            self.tables[b] = (o, c, length, cum)

    def to_clock(self, branch: str, ts: pd.Series) -> np.ndarray:
        o, c, length, cum = self.tables[branch]
        delta = (ts - self.ref).dt.total_seconds().to_numpy() / 60.0
        day = np.floor(delta / 1440.0).astype(int)
        minute = delta - day * 1440.0
        return cum[day] + np.clip(minute - o[day], 0.0, length[day])

    def from_clock(self, branch: str, clock: np.ndarray) -> pd.DatetimeIndex:
        o, c, length, cum = self.tables[branch]
        clock = np.asarray(clock, dtype=float)
        # first day whose open period ends at or after `clock` and that is open
        day = np.searchsorted(cum[1:], clock, side="left")
        day = np.clip(day, 0, len(o) - 1)
        minute = o[day] + (clock - cum[day])
        return self.ref + pd.to_timedelta(day * 1440.0 + minute, unit="min")

    def opening_clock(self, branch: str, ts: pd.Series) -> np.ndarray:
        """Clock value at the start of the open period that `ts` falls in (or the next one)."""
        o, c, length, cum = self.tables[branch]
        clk = self.to_clock(branch, ts)
        day = np.searchsorted(cum[1:], clk, side="right")  # day whose open period contains clk
        day = np.clip(day, 0, len(o) - 1)
        return cum[day]


def _make_clock(branches, *dfs) -> OpenClock:
    ts = pd.concat([d.return_ts for d in dfs])
    return OpenClock(branches, ts.min(), ts.max())


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_ts"] = pd.to_datetime(df.return_ts)
    for c in FLAG_COLS:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    for t in TASKS:
        df[t] = df[t].astype(int)
    return df


def _task_matrix(df: pd.DataFrame, branches: pd.DataFrame, branch_levels: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in NUM_COLS + FLAG_COLS:
        X[c] = df[c].astype(float)
    X["staff_on_duty"] = df.branch.map(branches.groupby("branch").staff_on_duty.first()).astype(float)
    for t in TASKS:
        X["flag_" + t] = df[t].astype(float)
    X["branch"] = df.branch.map({b: i for i, b in enumerate(branch_levels)}).astype(float)
    for c, lv in CAT_LEVELS.items():
        X[c] = df[c].map({v: i for i, v in enumerate(lv)}).astype(float)
    return X


TASK_CATS = ["branch"] + list(CAT_LEVELS)


def lpt_makespan(durations: np.ndarray, staff: np.ndarray) -> np.ndarray:
    """Longest-processing-time-first list scheduling of each row's tasks on `staff` workers."""
    d = -np.sort(-durations, axis=1)
    n = len(d)
    kmax = int(staff.max()) if n else 1
    loads = np.zeros((n, kmax))
    loads[np.arange(kmax)[None, :] >= staff[:, None]] = np.inf  # workers that do not exist
    rows = np.arange(n)
    for j in range(d.shape[1]):
        i = np.argmin(loads, axis=1)
        loads[rows, i] += d[:, j]
    loads[np.isinf(loads)] = 0.0
    return loads.max(axis=1) if n else np.zeros(0)


def _queue_features(df: pd.DataFrame, clock: OpenClock) -> pd.DataFrame:
    """Features describing the branch's arrival history before each car (open-minute clock)."""
    out = []
    windows = [10, 20, 30, 45, 60, 90, 120, 180]
    services = [10, 20, 30, 45]
    for b, g in df.groupby("branch", sort=True):
        k = clock.staff[b]
        c = clock.to_clock(b, g.return_ts)
        open_c = clock.opening_clock(b, g.return_ts)
        order = np.lexsort((g.turnaround_id.to_numpy(), g.return_ts.to_numpy(), c))
        cs = c[order]
        f = pd.DataFrame(index=g.index[order])
        f["clock"] = cs
        f["closed_return"] = (cs - clock.to_clock(b, g.return_ts.iloc[order] - pd.Timedelta(seconds=1)) == 0) & (
            np.isclose(cs, open_c[order]))
        f["min_since_open"] = cs - open_c[order]
        idx = np.arange(len(cs))
        for w in windows:
            f[f"prev{w}"] = idx - np.searchsorted(cs, cs - w, side="left")
        # cars ahead that arrived while the branch was closed and start at the same opening
        f["same_open_ahead"] = idx - np.searchsorted(cs, cs, side="left")
        for s in services:
            # single queue served by k workers, constant service time s per car
            free = np.full(k, -np.inf)
            waits = np.empty(len(cs))
            nsys = np.empty(len(cs))
            departs: list[float] = []
            for i, ci in enumerate(cs):
                j = int(np.argmin(free))
                start = max(ci, free[j])
                waits[i] = start - ci
                free[j] = start + s
                departs = [x for x in departs if x > ci]
                nsys[i] = len(departs)
                departs.append(start + s)
            f[f"sim_wait_s{s}"] = waits
            f[f"sim_nsys_s{s}"] = nsys * 12.0 / k
        out.append(f)
    return pd.concat(out).loc[df.index]


def _wait_matrix(df, branches, branch_levels, clock, ms_staff, chg) -> pd.DataFrame:
    q = _queue_features(df, clock)
    X = q.drop(columns=["clock"]).astype(float)
    X["staff_on_duty"] = df.branch.map(clock.staff).astype(float)
    X["branch"] = df.branch.map({b: i for i, b in enumerate(branch_levels)}).astype(float)
    X["return_hour"] = df.return_hour.astype(float)
    X["weekday"] = df.weekday.astype(float)
    X["queue_length"] = df.queue_length.astype(float)
    X["ms_staff"] = ms_staff
    X["charging"] = chg
    X["chg_minus_staff"] = chg - ms_staff
    return X


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------
def _hgb(cat_mask, n_rows, **kw):
    params = dict(
        loss="absolute_error", learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
        min_samples_leaf=max(5, min(40, n_rows // 50)), l2_regularization=1.0,
        early_stopping=False, categorical_features=cat_mask, random_state=SEED,
    )
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def _find_data_dir() -> Path | None:
    env = os.environ.get("TTD_DATA_DIR")
    here = Path(__file__).resolve().parent
    for p in ([Path(env)] if env else []) + [here.parent / "data", here.parent, Path.cwd() / "data", Path.cwd()]:
        if (p / "validation_labels.csv").exists() and (p / "validation.csv").exists():
            return p
    return None


def load_task_labels(data_dir: Path | None = None) -> pd.DataFrame:
    """Validation turnarounds joined with their true per-task durations (`true_<task>` columns)."""
    data_dir = Path(data_dir) if data_dir else _find_data_dir()
    if data_dir is None:
        raise FileNotFoundError("validation.csv / validation_labels.csv not found; set TTD_DATA_DIR")
    va = pd.read_csv(data_dir / "validation.csv")
    lab = pd.read_csv(data_dir / "validation_labels.csv")
    lab = lab.rename(columns={t: "true_" + t for t in TASKS}).drop(columns=["total_turnaround_min"])
    return va.merge(lab, on="turnaround_id", how="inner")


class GroupMedian:
    """Median duration per group (e.g. per branch staffing level), global median as fallback."""

    def __init__(self, key: str):
        self.key = key

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.medians_ = y.groupby(X[self.key].to_numpy()).median().to_dict()
        self.global_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.key].map(self.medians_).fillna(self.global_).to_numpy(dtype=float)


# Tyre checks are rare (about 1% of cars, 29 in the labelled week) and their duration is mostly set by
# branch size (about 15-20 min with 3 staff, 30-40 with 2, about 45 with 1), so a group median beats trees.
GROUP_MEDIAN_TASKS = {"tyre_check": "staff_on_duty"}


def _fit_task_models(labeled: pd.DataFrame, branches, branch_levels):
    labeled = _prepare(labeled)
    X = _task_matrix(labeled, branches, branch_levels)
    cat_mask = [c in TASK_CATS for c in X.columns]
    models, bounds = {}, {}
    for t in TASKS:
        rows = labeled[t] == 1
        y = labeled.loc[rows, "true_" + t].astype(float)
        if t in GROUP_MEDIAN_TASKS:
            mdl = GroupMedian(GROUP_MEDIAN_TASKS[t])
        elif rows.sum() >= 150:
            mdl = _hgb(cat_mask, int(rows.sum()))
        else:  # rare task: shallow trees, strong regularisation (refined later from train totals)
            mdl = _hgb(cat_mask, int(rows.sum()), max_iter=150, max_leaf_nodes=4, min_samples_leaf=8,
                       l2_regularization=5.0)
        mdl.fit(X[rows], y)
        models[t] = mdl
        bounds[t] = (float(y.min()), float(y.max()))
    return models, bounds


def _implied_durations(task, df, durs, target, staff, chg) -> np.ndarray:
    """Back out `task`'s duration from the observed time: the value r for which
    max(LPT(other staff tasks + r), charging) equals `target` (open minutes minus predicted wait).
    NaN where the task was not done or did not set the finish time (censored)."""
    others = durs[["dur_" + t for t in STAFF_TASKS if t != task]].to_numpy()

    def makespan(r, o, k, c):
        return np.maximum(lpt_makespan(np.column_stack([o, r]), k), c)

    out = np.full(len(df), np.nan)
    rows = (df[task].to_numpy() == 1) & (makespan(np.zeros(len(df)), others, staff, chg) < target - 0.5)
    if not rows.any():
        return out
    o, k, c, tgt = others[rows], staff[rows], chg[rows], target[rows]
    lo, hi = np.zeros(rows.sum()), np.full(rows.sum(), 400.0)
    for _ in range(40):  # bisection; makespan is non-decreasing in r
        mid = (lo + hi) / 2
        below = makespan(mid, o, k, c) < tgt
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    out[rows] = lo
    return out


def _predict_tasks(model, df, branches) -> pd.DataFrame:
    X = _task_matrix(df, branches, model["branch_levels"])
    out = pd.DataFrame(index=df.index)
    for t in TASKS:
        lo, hi = model["task_bounds"][t]
        p = np.clip(model["task_models"][t].predict(X), lo, hi)
        out["dur_" + t] = np.where(df[t] == 1, p, 0.0)
    return out


def _makespans(df, durs: pd.DataFrame, staff_map) -> tuple[np.ndarray, np.ndarray]:
    staff = df.branch.map(staff_map).to_numpy().astype(int)
    ms = lpt_makespan(durs[["dur_" + t for t in STAFF_TASKS]].to_numpy(), staff)
    return ms, durs["dur_electric_charging"].to_numpy()


def fit(train: pd.DataFrame, branches: pd.DataFrame, task_labels: pd.DataFrame | None = None):
    """Fit per-task models (labelled validation week) and the waiting-time model (train totals)."""
    if task_labels is None:
        task_labels = load_task_labels()
    train = _prepare(train).reset_index(drop=True)
    branch_levels = sorted(branches.branch.unique())
    task_models, task_bounds = _fit_task_models(task_labels, branches, branch_levels)
    model = {"branch_levels": branch_levels, "task_models": task_models, "task_bounds": task_bounds}

    # 1) waiting-time model: open minutes until ready, minus the car's own predicted makespan
    durs = _predict_tasks(model, train, branches)
    clock = _make_clock(branches, train)
    ms, chg = _makespans(train, durs, clock.staff)
    open_total = np.empty(len(train))
    for b, g in train.groupby("branch"):
        ready = g.return_ts + pd.to_timedelta(g.total_turnaround_min, unit="min")
        open_total[train.index.get_indexer(g.index)] = clock.to_clock(b, ready) - clock.to_clock(b, g.return_ts)
    Xw = _wait_matrix(train, branches, branch_levels, clock, ms, chg)
    wcat = [c == "branch" for c in Xw.columns]
    wait_model = _hgb(wcat, len(Xw), max_iter=500, max_leaf_nodes=31, min_samples_leaf=40)
    wait_model.fit(Xw, open_total - np.maximum(ms, chg))
    model["wait_model"] = wait_model

    # 2) quick damage repair: only ~70 labelled examples, but ~1,100 in train. Its duration there is
    #    backed out of the total (open minutes - predicted wait - the other tasks' predicted schedule).
    t = "quick_damage_repair"
    staff = train.branch.map(clock.staff).to_numpy().astype(int)
    excess = np.clip(wait_model.predict(Xw), 0.0, None)
    implied = _implied_durations(t, train, durs, open_total - excess, staff, chg)
    ok = ~np.isnan(implied)
    lab = _prepare(task_labels)
    lab = lab[lab[t] == 1]
    X = pd.concat([_task_matrix(train[ok], branches, branch_levels), _task_matrix(lab, branches, branch_levels)])
    y = np.concatenate([implied[ok], lab["true_" + t].to_numpy(dtype=float)])
    cat_mask = [c in TASK_CATS for c in X.columns]
    model["task_models"][t] = _hgb(cat_mask, len(y), max_iter=150, max_leaf_nodes=4, min_samples_leaf=20,
                                   l2_regularization=5.0).fit(X, y)
    model["task_bounds"][t] = (float(np.min(y)), float(np.max(y)))
    return model


def predict(model, df: pd.DataFrame, branches: pd.DataFrame) -> pd.DataFrame:
    df = _prepare(df).reset_index(drop=True)
    durs = _predict_tasks(model, df, branches)
    clock = _make_clock(branches, df)
    ms, chg = _makespans(df, durs, clock.staff)
    Xw = _wait_matrix(df, branches, model["branch_levels"], clock, ms, chg)
    excess = np.clip(model["wait_model"].predict(Xw), 0.0, None)
    open_need = np.maximum(ms, chg) + excess
    total = np.empty(len(df))
    for b, g in df.groupby("branch"):
        idx = df.index.get_indexer(g.index)
        start = clock.to_clock(b, g.return_ts)
        ready = clock.from_clock(b, start + open_need[idx])
        total[idx] = (ready.values - g.return_ts.values) / np.timedelta64(1, "m")
    out = pd.DataFrame({"turnaround_id": df.turnaround_id, "total_turnaround_min": np.round(total, 2)})
    for t in TASKS:
        out["dur_" + t] = np.round(durs["dur_" + t].to_numpy(), 2)
    return out
