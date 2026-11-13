"""Fit on the training data and write predictions.

Submission (run from starter/):

    mkdir -p ../submission && python run.py --test ../data/test.csv --out ../submission/test_predictions.csv
    python run.py --compare            # re-runs on ../data/test.csv and checks ../submission/test_predictions.csv

Other uses:

    python run.py                      # predicts validation.csv -> predictions.csv
    python run.py --repeat             # also checks that two runs give identical values
    python run.py --test other.csv --out other_predictions.csv
    python run.py --compare some_predictions.csv --test some_input.csv

Data is read from --data-dir (default: ../data if it exists, else the repo root).
Per-task models learn from validation.csv + validation_labels.csv (the only per-task truth),
so predictions *on the validation week itself* are in-sample; use validate_cv.py for an honest
leave-one-day-out estimate.
"""
import argparse
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import solution

HERE = Path(__file__).resolve().parent
SUBMISSION_FILE = HERE.parent / "submission" / "test_predictions.csv"


def default_data_dir() -> Path:
    for p in (HERE.parent / "data", HERE.parent):
        if (p / "train.csv").exists():
            return p
    raise FileNotFoundError("train.csv not found in ../data or the repo root; pass --data-dir")


def run_once(data_dir: Path, input_path: Path) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv")
    branches = pd.read_csv(data_dir / "branch_data.csv")
    df = pd.read_csv(input_path)
    model = solution.fit(train, branches)
    return solution.predict(model, df, branches)


def compare(pred: pd.DataFrame, saved_path: Path) -> bool:
    """True when `pred`, written as CSV, gives exactly the values stored in `saved_path`."""
    saved = pd.read_csv(saved_path)
    fresh = pd.read_csv(io.StringIO(pred.to_csv(index=False)))  # same CSV round trip as the saved file
    if list(saved.columns) != list(fresh.columns) or len(saved) != len(fresh):
        print(f"different shape or columns: saved {saved.shape} {list(saved.columns)[:3]}..., "
              f"new {fresh.shape} {list(fresh.columns)[:3]}...")
        return False
    if not saved.turnaround_id.equals(fresh.turnaround_id):
        print("turnaround_id order differs")
        return False
    num = [c for c in saved.columns if c != "turnaround_id"]
    diff = np.abs(saved[num].to_numpy(float) - fresh[num].to_numpy(float))
    n_diff = int((diff > 0).sum())
    print(f"compared {len(saved)} rows x {len(num)} values: {n_diff} differ, largest difference {diff.max():.4f} min")
    return n_diff == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--test", "--input", dest="test", type=Path, default=None,
                    help="file to predict (default: <data-dir>/validation.csv, or <data-dir>/test.csv with --compare)")
    ap.add_argument("--out", type=Path, default=None, help="where to write predictions (default: predictions.csv)")
    ap.add_argument("--repeat", action="store_true", help="run twice and check identical output")
    ap.add_argument("--compare", type=Path, nargs="?", const=SUBMISSION_FILE, default=None,
                    help=f"re-run and check that the predictions equal this file (default: {SUBMISSION_FILE})")
    args = ap.parse_args()
    data_dir = args.data_dir or default_data_dir()
    os.environ.setdefault("TTD_DATA_DIR", str(data_dir))
    input_path = args.test or data_dir / ("test.csv" if args.compare else "validation.csv")

    t0 = time.time()
    pred = run_once(data_dir, input_path)
    print(f"fit + predict on {input_path}: {time.time() - t0:.1f}s, {len(pred)} rows")
    out = args.out or (None if args.compare else Path("predictions.csv"))
    if out is not None:
        pred.to_csv(out, index=False)
        print(f"wrote {out}")
    if args.repeat:
        same = pred.equals(run_once(data_dir, input_path))
        print("repeat run identical:", same)
        if not same:
            raise SystemExit(1)
    if args.compare:
        ok = compare(pred, args.compare)
        print(f"reproduces {args.compare}: {ok}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
