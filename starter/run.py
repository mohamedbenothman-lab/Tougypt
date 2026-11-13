"""Fit on the training data and write predictions.

    python run.py                      # predicts validation.csv -> predictions.csv
    python run.py --repeat             # also checks that two runs give identical values
    python run.py --input other.csv --out other_predictions.csv

Data is read from --data-dir (default: ../data if it exists, else the repo root).
Per-task models learn from validation.csv + validation_labels.csv (the only per-task truth),
so predictions *on the validation week itself* are in-sample; use validate_cv.py for an honest
leave-one-day-out estimate.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd

import solution

HERE = Path(__file__).resolve().parent


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--input", type=Path, default=None, help="default: <data-dir>/validation.csv")
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--repeat", action="store_true", help="run twice and check identical output")
    args = ap.parse_args()
    data_dir = args.data_dir or default_data_dir()
    os.environ.setdefault("TTD_DATA_DIR", str(data_dir))
    input_path = args.input or data_dir / "validation.csv"

    t0 = time.time()
    pred = run_once(data_dir, input_path)
    print(f"fit + predict: {time.time() - t0:.1f}s, {len(pred)} rows")
    pred.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    if args.repeat:
        pred2 = run_once(data_dir, input_path)
        same = pred.equals(pred2)
        print("repeat run identical:", same)
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
