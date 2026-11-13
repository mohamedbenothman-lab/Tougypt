"""Honest validation: leave-one-day-out over the labelled validation week.

For each validation day, the whole model is refitted with the labels of the other six days only:
per-task models, the waiting-time model (whose inputs use those task models) and the repair model
backed out of train totals. Each day is then predicted by a model that never saw its labels.

    python validate_cv.py            # writes predictions_cv.csv and prints the scores
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd

import solution
from evaluate_local import evaluate
from run import default_data_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("predictions_cv.csv"))
    args = ap.parse_args()
    data_dir = args.data_dir or default_data_dir()
    os.environ.setdefault("TTD_DATA_DIR", str(data_dir))

    train = pd.read_csv(data_dir / "train.csv")
    branches = pd.read_csv(data_dir / "branch_data.csv")
    va = pd.read_csv(data_dir / "validation.csv")
    labels = pd.read_csv(data_dir / "validation_labels.csv")
    task_labels = solution.load_task_labels(data_dir)

    day = pd.to_datetime(va.return_ts).dt.date
    lab_day = pd.to_datetime(task_labels.return_ts).dt.date
    parts = []
    for d in sorted(day.unique()):
        t0 = time.time()
        fold = solution.fit(train, branches, task_labels[lab_day != d])
        pred = solution.predict(fold, va, branches)  # whole week, so queue features see every arrival
        parts.append(pred[(day == d).to_numpy()])
        print(f"fold {d}: {time.time() - t0:.1f}s")
    pred = pd.concat(parts).set_index("turnaround_id").loc[va.turnaround_id].reset_index()
    pred.to_csv(args.out, index=False)
    print(f"wrote {args.out}\n")
    evaluate(pred, labels, va)


if __name__ == "__main__":
    main()
