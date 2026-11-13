"""Task Time Detective: per-task durations and total turnaround time.

How the data works (found by checking the validation labels; see SUBMISSION.md):

* Every task has its own duration, and the effects add up (a child seat adds the same ~7.5 min of
  interior cleaning at every dirt level). Frequent tasks use additive median regressions learned from
  the labelled validation week; quick damage repair also learns from durations backed out of the
  `train.csv` totals, and tyre checks use a median per branch staffing level.
* The staff tasks of one car are split across the branch's `staff_on_duty` workers by a fixed rule
  (see `staff_schedule`): a repair comes first (after the damage check, with the car alone), the other
  tasks are handed out in a fixed order to whichever worker is free first, and interior cleaning waits
  for special cleaning. Electric charging runs in parallel on the charger.
  makespan = max(staff schedule, charging)
* Before work starts, a car waits for the cars already queued at the branch, roughly 12 min / staff
  per car ahead. Service times in that queue are random, so the wait is estimated from the branch's
  recent arrivals and how close closing time is (a model trained on the `train.csv` totals).
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
from sklearn.linear_model import QuantileRegressor

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

    def closing(self, branch: str, clk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For clock values: the clock at the end of their open period (closing time), and how many
        calendar minutes the branch then stays closed (a night, or a whole weekend at city centers)."""
        o, c, length, cum = self.tables[branch]
        day = np.clip(np.searchsorted(cum[1:], clk, side="right"), 0, len(o) - 1)
        end = cum[day + 1]
        close_t = self.from_clock(branch, end)
        reopen_t = self.from_clock(branch, end + 1e-6)
        return end, (reopen_t - close_t).total_seconds().to_numpy() / 60.0


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


# Within-car schedule, found in the validation labels (it reproduces 97-98% of the totals of cars that
# arrived at a quiet branch exactly, against 82% for longest-task-first):
#  * a quick damage repair comes first: damage check, then the repair, with the car alone;
#  * the other staff tasks are then handed out in this fixed order, each to whichever worker is free first;
#  * interior cleaning waits until special cleaning is finished (both happen inside the car);
#  * electric charging runs alongside on the charger (handled by the caller).
TASK_ORDER = ["exterior", "special_cleaning", "tyre_check", "interior_cleaning", "damage_check", "refuel"]


def staff_schedule(d: dict, staff: np.ndarray) -> np.ndarray:
    """Minutes until the staff tasks of each car are done. `d` maps task name -> durations (0 = not done)."""
    n = len(staff)
    if n == 0:
        return np.zeros(0)
    repair, check = d["quick_damage_repair"], d["damage_check"]
    t0 = np.where(repair > 0, check + repair, 0.0)
    jobs = {
        "exterior": d["quick_exterior_cleaning"] + d["normal_exterior_cleaning"],
        "special_cleaning": d["special_cleaning"], "tyre_check": d["tyre_check"],
        "interior_cleaning": d["interior_cleaning"], "damage_check": np.where(repair > 0, 0.0, check),
        "refuel": d["refuel"],
    }
    kmax = int(staff.max())
    free = np.repeat(t0[:, None], kmax, axis=1)
    absent = np.arange(kmax)[None, :] >= staff[:, None]  # workers this branch does not have
    free[absent] = np.inf
    rows = np.arange(n)
    special_done = np.zeros(n)
    for t in TASK_ORDER:
        dur = jobs[t]
        j = np.argmin(free, axis=1)
        start = free[rows, j]
        if t == "interior_cleaning":
            start = np.maximum(start, special_done)
        end = start + dur
        free[rows, j] = np.where(dur > 0, end, free[rows, j])
        if t == "special_cleaning":
            special_done = np.where(dur > 0, end, 0.0)
    free[absent] = -np.inf
    return free.max(axis=1)


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
    # closing time: a car whose work may run past closing loses a whole night (or weekend), and
    # end-of-day queues behave differently, so the wait model sees how close closing is
    start = q["clock"].to_numpy()
    end, closure = np.empty(len(df)), np.empty(len(df))
    for b, g in df.groupby("branch"):
        idx = df.index.get_indexer(g.index)
        end[idx], closure[idx] = clock.closing(b, start[idx])
    X["min_to_close"] = end - start
    X["slack_before_wait"] = end - start - np.maximum(ms_staff, chg)
    X["closure_min"] = closure
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
    """Validation turnarounds joined with their true per-task durations (`true_<task>` columns) and
    their true `total_turnaround_min`."""
    data_dir = Path(data_dir) if data_dir else _find_data_dir()
    if data_dir is None:
        raise FileNotFoundError("validation.csv / validation_labels.csv not found; set TTD_DATA_DIR")
    va = pd.read_csv(data_dir / "validation.csv")
    lab = pd.read_csv(data_dir / "validation_labels.csv")
    lab = lab.rename(columns={t: "true_" + t for t in TASKS})
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


def _linear_matrix(df: pd.DataFrame, branch_levels: list[str]) -> pd.DataFrame:
    """Additive design: numbers as they are, every category level as its own 0/1 column."""
    Z = pd.DataFrame(index=df.index)
    for c in ["rental_days", "km_driven", "return_hour", "queue_length"]:
        Z[c] = df[c].astype(float)
    for c in ["fuel_level_pct", "battery_soc_pct"]:
        Z[c] = df[c].astype(float).fillna(0.0)
        Z[c + "_missing"] = df[c].isna().astype(float)
    for c in ["interior_dirt", "exterior_dirt"]:
        for v in [2, 3, 4, 5]:
            Z[f"{c}={v}"] = (df[c] == v).astype(float)
    for c in FLAG_COLS:
        Z[c] = df[c].astype(float)
    for t in TASKS:
        if t != "normal_exterior_cleaning":  # exactly one exterior cleaning: the other flag says it all
            Z["flag_" + t] = df[t].astype(float)
    for c, lv in [("branch", branch_levels)] + list(CAT_LEVELS.items()):
        for v in lv[1:]:
            Z[f"{c}={v}"] = (df[c] == v).astype(float)
    for w in range(1, 7):
        Z[f"weekday={w}"] = (df.weekday == w).astype(float)
    return Z


class MedianLinear:
    """Additive median (absolute-error) regression on standardised columns. The task durations turn out
    to be additive (e.g. a child seat adds the same ~7.5 min to interior cleaning at every dirt level),
    which this captures with far fewer parameters than trees need."""

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    def fit(self, Z: pd.DataFrame, y):
        sd = Z.std()
        self.cols_ = list(Z.columns[sd > 0])
        self.mu_, self.sd_ = Z[self.cols_].mean(), Z[self.cols_].std()
        A = ((Z[self.cols_] - self.mu_) / self.sd_).to_numpy()
        self.q_ = QuantileRegressor(quantile=0.5, alpha=self.alpha, solver="highs").fit(A, np.asarray(y, float))
        return self

    def predict(self, Z: pd.DataFrame) -> np.ndarray:
        return self.q_.predict(((Z[self.cols_] - self.mu_) / self.sd_).to_numpy())


class TaskModel:
    """One task's model: 'linear' (additive median regression), 'trees' (gradient boosting) or
    'group' (median per group)."""

    def __init__(self, kind: str, trees=None, linear=None, group=None):
        self.kind, self.trees, self.linear, self.group = kind, trees, linear, group

    def predict(self, X: pd.DataFrame, Z: pd.DataFrame) -> np.ndarray:
        if self.kind == "linear":
            return self.linear.predict(Z)
        if self.kind == "trees":
            return self.trees.predict(X)
        return self.group.predict(X)


# Chosen per task by leave-one-day-out MAE on the labelled week (see SUBMISSION.md).
# Frequent tasks are additive, so a median linear model is used; the L1 penalty is larger for tasks
# driven by one or two inputs (charging is ~40 - 0.35 min per % of battery and nothing else matters).
# Tyre checks are rare (29 labelled) and mostly set by branch size, so a median per staffing level wins.
# Quick damage repair starts as trees here and is refitted from train totals in fit().
TASK_KIND = {
    "quick_exterior_cleaning": "linear", "normal_exterior_cleaning": "linear", "interior_cleaning": "linear",
    "damage_check": "linear", "refuel": "linear", "special_cleaning": "linear", "electric_charging": "linear",
    "quick_damage_repair": "trees", "tyre_check": "group",
}
LINEAR_ALPHA = {
    "quick_exterior_cleaning": 0.003, "normal_exterior_cleaning": 0.001, "interior_cleaning": 0.003,
    "damage_check": 0.01, "refuel": 0.001, "special_cleaning": 0.01, "electric_charging": 0.04,
}
GROUP_KEY = {"tyre_check": "staff_on_duty"}


def _fit_task_models(labeled: pd.DataFrame, branches, branch_levels):
    labeled = _prepare(labeled)
    X = _task_matrix(labeled, branches, branch_levels)
    Z = _linear_matrix(labeled, branch_levels)
    cat_mask = [c in TASK_CATS for c in X.columns]
    models, bounds = {}, {}
    for t in TASKS:
        rows = (labeled[t] == 1).to_numpy()
        y = labeled.loc[rows, "true_" + t].astype(float)
        kind = TASK_KIND[t]
        trees = linear = group = None
        if kind == "group":
            group = GroupMedian(GROUP_KEY[t]).fit(X[rows], y)
        if kind == "linear":
            linear = MedianLinear(LINEAR_ALPHA[t]).fit(Z[rows], y)
        if kind == "trees":
            if rows.sum() >= 150:
                trees = _hgb(cat_mask, int(rows.sum()))
            else:  # rare task: shallow trees, strong regularisation
                trees = _hgb(cat_mask, int(rows.sum()), max_iter=150, max_leaf_nodes=4, min_samples_leaf=8,
                             l2_regularization=5.0)
            trees.fit(X[rows], y)
        models[t] = TaskModel(kind, trees, linear, group)
        bounds[t] = (float(y.min()), float(y.max()))
    return models, bounds


def _implied_durations(task, df, durs, target, staff, chg) -> np.ndarray:
    """Back out `task`'s duration from the observed time: the value r for which
    max(staff schedule with `task` = r, charging) equals `target` (open minutes minus predicted wait).
    NaN where the task was not done or did not set the finish time (censored)."""
    base = {t: durs["dur_" + t].to_numpy() for t in TASKS}

    def makespan(r, idx):
        d = {t: v[idx] for t, v in base.items()}
        d[task] = r
        return np.maximum(staff_schedule(d, staff[idx]), chg[idx])

    out = np.full(len(df), np.nan)
    every = np.arange(len(df))
    rows = (df[task].to_numpy() == 1) & (makespan(np.zeros(len(df)), every) < target - 0.5)
    if not rows.any():
        return out
    idx, tgt = np.flatnonzero(rows), target[rows]
    lo, hi = np.zeros(len(idx)), np.full(len(idx), 400.0)
    for _ in range(40):  # bisection; the schedule is non-decreasing in r
        mid = (lo + hi) / 2
        below = makespan(mid, idx) < tgt
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    out[rows] = lo
    return out


def _predict_tasks(model, df, branches) -> pd.DataFrame:
    X = _task_matrix(df, branches, model["branch_levels"])
    Z = _linear_matrix(df, model["branch_levels"])
    out = pd.DataFrame(index=df.index)
    for t in TASKS:
        lo, hi = model["task_bounds"][t]
        p = np.clip(model["task_models"][t].predict(X, Z), lo, hi)
        out["dur_" + t] = np.where(df[t] == 1, p, 0.0)
    return out


def _makespans(df, durs: pd.DataFrame, staff_map) -> tuple[np.ndarray, np.ndarray]:
    staff = df.branch.map(staff_map).to_numpy().astype(int)
    ms = staff_schedule({t: durs["dur_" + t].to_numpy() for t in TASKS}, staff)
    return ms, durs["dur_electric_charging"].to_numpy()


def _wait_rows(model, df: pd.DataFrame, branches, branch_levels):
    """Wait-model inputs and target for a period with known totals: predicted task times, the open-minute
    clock, the car's predicted makespan, and open minutes from return until ready."""
    durs = _predict_tasks(model, df, branches)
    clock = _make_clock(branches, df)
    ms, chg = _makespans(df, durs, clock.staff)
    open_total = np.empty(len(df))
    for b, g in df.groupby("branch"):
        ready = g.return_ts + pd.to_timedelta(g.total_turnaround_min, unit="min")
        open_total[df.index.get_indexer(g.index)] = clock.to_clock(b, ready) - clock.to_clock(b, g.return_ts)
    Xw = _wait_matrix(df, branches, branch_levels, clock, ms, chg)
    return durs, clock, ms, chg, open_total, Xw


def fit(train: pd.DataFrame, branches: pd.DataFrame, task_labels: pd.DataFrame | None = None):
    """Fit per-task models (labelled validation week) and the waiting-time model (train totals, plus the
    labelled week's totals when they are given)."""
    if task_labels is None:
        task_labels = load_task_labels()
    train = _prepare(train).reset_index(drop=True)
    branch_levels = sorted(branches.branch.unique())
    task_models, task_bounds = _fit_task_models(task_labels, branches, branch_levels)
    model = {"branch_levels": branch_levels, "task_models": task_models, "task_bounds": task_bounds}

    # 1) waiting-time model: open minutes until ready, minus the car's own predicted makespan.
    #    The labelled week is the period closest to the test weeks, so its totals are added too; its
    #    queue features are computed within that week, like they will be for a test period.
    durs, clock, ms, chg, open_total, Xw = _wait_rows(model, train, branches, branch_levels)
    X_all, y_all = [Xw], [open_total - np.maximum(ms, chg)]
    if "total_turnaround_min" in task_labels.columns:
        lab = _prepare(task_labels).reset_index(drop=True)
        lab = lab[lab.total_turnaround_min.notna()].reset_index(drop=True)
        if len(lab):
            _, _, ms_l, chg_l, ot_l, Xw_l = _wait_rows(model, lab, branches, branch_levels)
            X_all.append(Xw_l)
            y_all.append(ot_l - np.maximum(ms_l, chg_l))
    X_all, y_all = pd.concat(X_all, ignore_index=True), np.concatenate(y_all)
    wcat = [c == "branch" for c in X_all.columns]
    wait_model = _hgb(wcat, len(X_all), max_iter=500, max_leaf_nodes=31, min_samples_leaf=40)
    wait_model.fit(X_all, y_all)
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
    trees = _hgb(cat_mask, len(y), max_iter=150, max_leaf_nodes=4, min_samples_leaf=20,
                 l2_regularization=5.0).fit(X, y)
    model["task_models"][t] = TaskModel("trees", trees=trees)
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
