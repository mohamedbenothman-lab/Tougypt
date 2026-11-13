"""Local re-implementation of the README's scoring (the official evaluate.py is not in this repo).

    python evaluate_local.py predictions.csv --labels ../data/validation_labels.csv --flags ../data/validation.csv
"""
import argparse

import numpy as np
import pandas as pd

TASKS = [
    "quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
    "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check",
]


def evaluate(pred: pd.DataFrame, labels: pd.DataFrame, flags: pd.DataFrame, verbose: bool = True) -> dict:
    missing = [c for c in ["total_turnaround_min"] + ["dur_" + t for t in TASKS] if c not in pred.columns]
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    m = flags[["turnaround_id"] + TASKS].merge(labels, on="turnaround_id", suffixes=("_flag", ""))
    m = m.merge(pred.rename(columns={"total_turnaround_min": "pred_total"}), on="turnaround_id", how="left")
    if m["pred_total"].isna().any():
        raise ValueError("predictions missing some turnaround_ids")
    true_total, pred_total = m["total_turnaround_min"], m["pred_total"]

    rows, abs_err_sum, true_sum = [], 0.0, 0.0
    for t in TASKS:
        done = m[t + "_flag"] == 1
        y, p = m.loc[done, t], m.loc[done, "dur_" + t]
        mae = (y - p).abs().mean()
        base = (y - labels.loc[labels[t] > 0, t].mean()).abs().mean()
        rows.append({"task": t, "n": int(done.sum()), "true_mean": y.mean(), "pred_mean": p.mean(),
                     "MAE": mae, "MAE_avg_guess": base, "skill": 1 - mae / base if base > 0 else np.nan})
        abs_err_sum += (y - p).abs().sum()
        true_sum += y.sum()
    per_task = pd.DataFrame(rows)
    durs = m[["dur_" + t for t in TASKS]]
    not_done = np.column_stack([(m[t + "_flag"] == 0) & (m["dur_" + t] != 0) for t in TASKS])
    res = {
        "per_task": per_task,
        "mean_task_MAE": per_task.MAE.mean(),
        "WAPE": abs_err_sum / true_sum,
        "total_MAE": (true_total - pred_total).abs().mean(),
        "n_negative_durations": int((durs < 0).sum().sum()),
        "n_nonzero_for_tasks_not_done": int(not_done.sum()),
        "n_total_below_longest_task": int((pred_total < durs.max(axis=1) - 1e-6).sum()),
    }
    if verbose:
        print(per_task.round(3).to_string(index=False))
        for k, v in res.items():
            if k != "per_task":
                print(f"{k:32s} {v:.4f}" if isinstance(v, float) else f"{k:32s} {v}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--flags", required=True)
    a = ap.parse_args()
    evaluate(pd.read_csv(a.predictions), pd.read_csv(a.labels), pd.read_csv(a.flags))


if __name__ == "__main__":
    main()
