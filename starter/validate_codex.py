"""Compare schedule-aware models with out-of-fold task and elapsed predictions."""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from linear_regression import TASKS, build_features, find_data_dir, opening_features, read_hours

KEY = "turnaround_id"
TOTAL = "total_turnaround_min"


def regressor():
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=85, max_leaf_nodes=15, random_state=0
    )


def task_features(frame):
    return pd.get_dummies(frame.drop(columns=[KEY, "return_ts"]), dtype=float)


def score(train, validation, labels, branches, forward):
    # Include training arrivals as history for the public week. Feature rows have
    # no labels, and the shared queue code orders arrivals by their return time.
    all_rows = pd.concat([train.drop(columns=TOTAL), validation], ignore_index=True)
    op = opening_features(all_rows.assign(return_ts=pd.to_datetime(all_rows.return_ts)),
                          read_hours(branches))
    days = pd.to_datetime(validation.return_ts).dt.normalize()
    labels = labels.set_index(KEY).loc[validation[KEY]].reset_index()
    task_x = task_features(validation)
    staff = validation.branch.map(branches.groupby("branch").staff_on_duty.first())
    n = len(train)
    y = train[TOTAL].to_numpy()
    predictions = []
    probability_rows = []
    for day in sorted(days.unique()):
        test = days.eq(day)
        fit = days.lt(day) if forward else ~test
        if not fit.any():
            continue
        typical = labels.loc[fit, TASKS].where(labels.loc[fit, TASKS] > 0).median()
        if typical.isna().any():
            # Compare on days with an earlier example of every task.
            continue
        x = build_features(all_rows, branches, typical)
        x["minutes_to_close"] = op.min_left_before_close
        x["next_gap"] = op.night_min
        x["queue_length"] = all_rows.queue_length
        x["predicted_minutes"] = x[["longest_job_min","work_min_per_staff"]].max(axis=1) + 12*x.cars_ahead_per_staff
        x["closing_margin"] = x.minutes_to_close - x.predicted_minutes
        x["work_start_minute"] = op.work_start.dt.hour*60 + op.work_start.dt.minute
        x["weekday"] = op.work_start.dt.weekday
        x = pd.concat([x, all_rows[TASKS], pd.get_dummies(all_rows.branch, dtype=float)],axis=1)
        wait = op.closed_wait_min.to_numpy()
        reopen = wait + op.min_left_before_close.to_numpy() + op.night_min.to_numpy()
        near = (wait == 0) & (op.min_left_before_close.to_numpy() <= 120) & (op.night_min.to_numpy() > 0)
        carry = near[:n] & (y >= reopen[:n])
        direct = regressor().fit(x.iloc[:n], y - wait[:n])
        classifier = HistGradientBoostingClassifier(
            max_iter=85, max_leaf_nodes=15, random_state=0
        ).fit(x.iloc[:n].loc[near[:n]], carry[near[:n]])
        same_shift = regressor().fit(x.iloc[:n].loc[near[:n] & ~carry], y[near[:n] & ~carry])
        after = regressor().fit(x.iloc[:n].loc[carry], y[carry] - reopen[:n][carry])
        i = n + np.flatnonzero(test)
        base = np.maximum(0, direct.predict(x.iloc[i]) + wait[i])
        candidate = base.copy()
        late = np.flatnonzero(near[i])
        probability = classifier.predict_proba(x.iloc[i[late]])[:, 1] if len(late) else np.empty(0)
        pre = np.maximum(0, same_shift.predict(x.iloc[i[late]])) if len(late) else np.empty(0)
        post = reopen[i[late]] + np.maximum(0, after.predict(x.iloc[i[late]])) if len(late) else np.empty(0)
        probability_rows.append(pd.DataFrame({KEY: validation.loc[test, KEY].to_numpy()[late],
            "probability": probability, "same": pre, "next": post}))
        selected = late[probability >= .5]
        same = late[probability < .5]
        if len(same):
            candidate[same] = np.maximum(0, same_shift.predict(x.iloc[i[same]]))
        if len(selected):
            candidate[selected] = reopen[i[selected]] + np.maximum(0, after.predict(x.iloc[i[selected]]))
        out = pd.DataFrame({
            KEY: validation.loc[test, KEY].to_numpy(), "day": day,
            "branch": validation.loc[test, "branch"].to_numpy(),
            "return_ts": validation.loc[test, "return_ts"].to_numpy(),
            "minutes_to_close": op.min_left_before_close.to_numpy()[i],
            "minutes_to_next_open": reopen[i], "actual_total": labels.loc[test, TOTAL].to_numpy(),
            "baseline_total": base, TOTAL: candidate,
            "group": np.where(wait[i]>0, "closed", np.where(op.min_left_before_close.to_numpy()[i]<=60, "last_hour", "earlier")),
        })
        out["actual_carry"] = near[i] & (out.actual_total >= reopen[i])
        out["predicted_carry"] = near[i] & (candidate >= reopen[i])
        for task in TASKS:
            actual = labels[task]
            performed = fit & actual.gt(0)
            active = validation.loc[test, task].to_numpy() == 1
            values = np.full(test.sum(), typical[task])
            if task == "tyre_check":
                medians = actual[performed].groupby(staff[performed]).median()
                values = staff[test].map(medians).fillna(typical[task]).to_numpy()
            elif performed.sum() >= 40:
                values = regressor().fit(task_x[performed], actual[performed]).predict(task_x[test])
            out["dur_"+task] = np.where(active, np.maximum(values, 0), 0)
            out["median_"+task] = np.where(active, typical[task], 0)
            out["actual_"+task] = actual[test].to_numpy()
        predictions.append(out)
    return pd.concat(predictions, ignore_index=True).merge(
        pd.concat(probability_rows, ignore_index=True), on=KEY, how="left", validate="one_to_one"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true", help="Fit task models using only earlier public days.")
    args = parser.parse_args()
    data = find_data_dir()
    train, val, labels, branches = [
        pd.read_csv(data / f"{name}.csv", dtype={KEY: str})
        for name in ("train", "validation", "validation_labels", "branch_data")
    ]
    assert val[KEY].is_unique and labels[KEY].is_unique and set(val[KEY]) == set(labels[KEY])
    out = score(train, val, labels, branches, args.forward)
    out["absolute_error"] = (out[TOTAL] - out.actual_total).abs()
    out["baseline_error"] = (out.baseline_total - out.actual_total).abs()
    print(out.groupby("group").agg(n=(KEY,"size"), baseline_mae=("baseline_error","mean"),
                                   candidate_mae=("absolute_error","mean")).round(3))
    print("Elapsed MAE (direct, spillover):", out[["baseline_error","absolute_error"]].mean().round(3).tolist())
    truth = out[["actual_"+t for t in TASKS]].to_numpy()
    mask = truth > 0
    for prefix in ("median_", "dur_"):
        err = np.abs(out[[prefix+t for t in TASKS]].to_numpy() - truth)
        mae = np.divide((err*mask).sum(axis=0), mask.sum(axis=0), out=np.full(len(TASKS),np.nan), where=mask.sum(axis=0)>0)
        print(prefix, dict(zip(TASKS, np.round(mae,3))), "mean MAE", np.nanmean(mae),
              "WAPE", err[mask].sum()/truth[mask].sum())
    suffix = "forward" if args.forward else "leave_one_day_out"
    folder = data.parent / "starter" if data.name == "data" else data / "starter"
    out.to_csv(folder / f"predictions_{suffix}.csv", index=False)
    out[out.group.eq("last_hour")].nlargest(16, "absolute_error").to_csv(folder / f"closing_misses_{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
