# Task Time Detective: submission

## Results (validation week, leave-one-day-out, so no model saw the day it predicts)

| Metric | Baseline (`solution.py` as given) | This model | README "Strong" |
|---|---|---|---|
| Per-task WAPE | ~0.55 | **0.078** | < 0.2 |
| Mean per-task MAE (all 9 tasks weighted equally) | – | **2.71 min** | – |
| Total turnaround MAE | ~99 min | **8.94 min** | < 5 min |
| Negative or impossible durations | – | 0 | – |

- The total is in the "Good" band (< 15 min). 5.0 of its 8.9 min come from **16 of 2,791 cars** whose
  work ends right around closing time, where a few minutes of queue decide between "ready tonight" and
  "ready tomorrow morning". Without those cars, the total MAE is 4.0 min (see Caveats).
- Two runs give byte-identical output (`run.py --repeat`, plus a separate process run). Fit + predict
  takes about 23 s on a laptop.

## How the turnaround time is built

Checked against the true per-task times in `validation_labels.csv`:

1. **Tasks on one car run in parallel across the staff.** The total equals the sum of task times only
   5% of the time, but it is never shorter than the longest task. The staff tasks are split across the
   branch's `staff_on_duty` workers, longest first (LPT list scheduling). With 1 staff (city centers),
   everything is sequential. **Electric charging runs on the charger in parallel** and needs no
   staff. This gives `makespan = max(LPT(staff tasks, staff_on_duty), charging)`, and it never
   exceeds the true total (0 violations on validation).
2. **Cars wait for the queue at the branch.** The remaining time is a wait of about **12 min ÷ staff
   for each car ahead**: about 12 min steps at 1-staff centers, 6 at stations, 4 at airports. The
   clearest case is the Monday-morning backlog at the centers, where the n-th car from the Sunday key
   box waits about n × 12 min. The service times in that queue are random and unobserved, so the wait
   can only be estimated from the branch's arrival history. `queue_length` is *not* related to it.
3. **Work only happens while the branch is open.** A car left in the key box starts at the next
   opening. Work still running at closing time pauses until the next opening (+480 min overnight at
   stations, +750 min at centers, a whole Sunday at centers). With an "open-minute" clock per branch,
   `ready = return + (wait + makespan)` counted in open minutes reproduces every validation total
   with a non-negative wait.

## Approach (`starter/solution.py`)

| Part | Model | Trained on |
|---|---|---|
| 7 frequent tasks | `HistGradientBoostingRegressor`, absolute-error loss, one per task | validation labels (true task minutes) |
| `tyre_check` | median per branch staffing level (1/2/3 staff) | validation labels (29 rows; trees overfit) |
| `quick_damage_repair` | small gradient-boosted trees | ~1,100 durations backed out of `train.csv` totals + the 71 validation labels |
| waiting time | gradient boosting, absolute-error loss | `train.csv` totals: open minutes until ready minus the car's predicted makespan |
| total | `return + open-minute clock(predicted makespan + predicted wait)` | assembled, no extra fitting |

- **Task features:** car condition (dirt scores, child seat, pet, smoking, damage), vehicle class,
  powertrain, fuel/battery level, km, rental days, branch, staff, weather, hour, weekday,
  `queue_length` and the other task flags. Predictions are clipped to the observed range, and a
  task that was not done gets 0.
- **Repair from totals:** only 71 repairs are labelled, but `train.csv` has ~1,100. For each one, the
  repair time is the value r that makes `max(LPT(other predicted tasks + r), charging)` equal the
  observed open minutes minus the predicted wait (solved by bisection). On validation, where the
  truth is known, these backed-out values are within ~5.5 min of the true repair time.
- **Wait features:** arrivals at the same branch in the previous 10–180 open minutes, cars waiting
  from the same closure (key-box backlog), minutes since opening, small deterministic queue
  simulations (k servers, fixed service times), staff, branch, hour, weekday, and the car's own
  makespan. Only past arrivals are used, never future ones.
- **Determinism:** fixed `random_state`, no early stopping, no sampling. `python run.py --repeat`
  checks that two runs are identical.

## Feature importance (permutation importance: MAE increase in minutes when a feature is shuffled)

Computed on the labelled week with the final models; top features per task:

| Task | Most important features | Does it make sense? |
|---|---|---|
| quick exterior cleaning | branch 0.36, exterior_dirt 0.23, queue_length 0.13, vehicle_class 0.11 | yes: dirtier and bigger cars take longer |
| normal exterior cleaning | exterior_dirt 1.32, branch 0.37, vehicle_class 0.37, weather 0.32 | yes: dirt score 1→5 adds ~7 min, rain adds ~1.5 |
| special cleaning | smoking_reported 1.41, vehicle_class 0.67, pet 0.42, child_seat 0.41 | yes: smoking adds ~7.6 min |
| interior cleaning | interior_dirt 1.64, child_seat 1.42, branch 0.32, vehicle_class 0.25 | yes: dirt 1→5 adds ~9.5 min, a child seat ~7.5 |
| damage check | nothing material (all < 0.05) | yes: a near-constant ~2.4 min check |
| refuel | branch 3.00, vehicle_class 1.13, km_driven 0.22, fuel_level_pct 0.18 | plausible: distance to the pump differs by branch, and tank size by class |
| electric charging | battery_soc_pct 7.57, km_driven 0.42, branch 0.31 | yes: the lower the battery, the longer the charge (correlation −0.96) |
| quick damage repair | damage_reported 5.81, vehicle_class 0.67, battery_soc_pct 0.50 | yes for damage (+~20 min); the rest is mostly unexplained noise |
| tyre check | group median by branch staffing: 47.0 / 33.3 / 17.5 min for 1 / 2 / 3 staff | plausible: small branches lack equipment and hands |

**Waiting-time model** (MAE increase, 6,000 training rows): return_hour 4.27, min_since_open 4.14,
ms_staff 3.63, prev30 (arrivals in the last 30 open minutes) 1.92, branch 1.45, same_open_ahead
(key-box backlog) 1.40, prev45 1.36, sim_wait_s20 1.30. The wait is driven by the time of day and
how many cars arrived just before. `queue_length` adds nothing.

## Per-task estimates (validation week, leave-one-day-out predictions)

| Task | Cars (val) | True mean (min) | Predicted mean | MAE | MAE if always guessing the average | Skill | Typical estimate |
|---|---|---|---|---|---|---|---|
| quick exterior cleaning | 1,575 | 13.3 | 13.2 | 0.83 | 1.02 | 0.19 | 12.6 (clean) → 15.0 (dirt 5) |
| normal exterior cleaning | 1,216 | 19.2 | 19.2 | 1.70 | 2.29 | 0.26 | 16.3 (clean) → 23.5 (dirt 5) |
| special cleaning | 277 | 30.0 | 30.0 | 0.86 | 2.11 | 0.59 | ~29; ~37 if smoking reported |
| interior cleaning | 2,791 | 16.4 | 16.4 | 1.31 | 2.96 | 0.56 | 13.3 (clean) → 22.8 (dirt 5); +7.5 with a child seat |
| damage check | 1,394 | 2.4 | 2.4 | 0.21 | 0.21 | 0.02 | ~2.4 (barely varies) |
| refuel | 410 | 13.5 | 13.5 | 0.90 | 2.86 | 0.68 | 7–17 depending on branch; larger cars longer |
| electric charging | 526 | 24.8 | 24.8 | 1.75 | 6.16 | 0.72 | set by battery level at return (3–40 min) |
| quick damage repair | 71 | 68.6 | 68.1 | 12.57 | 15.12 | 0.17 | ~55 without reported damage, ~76 with |
| tyre check | 29 | 26.1 | 25.7 | 4.25 | 10.28 | 0.59 | 17.5 / 33 / 47 at 3- / 2- / 1-staff branches |

Skill = 1 − MAE / MAE of always guessing the average. As the README warns, it is harsh for tasks
that barely vary, such as the damage check.

## How to run

```bash
pip install -r requirements.txt        # Python 3.11+
cd starter
python run.py --repeat                 # fit on train + labels, predict validation.csv -> predictions.csv, check determinism
python validate_cv.py                  # honest leave-one-day-out scores -> predictions_cv.csv
python evaluate_local.py predictions_cv.csv --labels ../validation_labels.csv --flags ../validation.csv
python run.py --input <test.csv> --out <test_predictions.csv>
```

The output has one row per turnaround: `turnaround_id`, `total_turnaround_min`, and `dur_<task>` for
all 9 tasks (0 when the task was not done).

## Caveats

- **Data location:** the CSVs sit at the repo root, not in `data/`. `run.py` looks in `../data` first,
  then the repo root (override with `--data-dir` or `TTD_DATA_DIR`).
- **Missing organiser files:** the official `evaluate.py`, `run.py` template and `setup.sh` are not
  in this repo. `evaluate_local.py` re-implements the README's metrics, and the output columns above
  are a best guess at the official format.
- **`fit(train, branches)` also reads the validation labels** (`validation.csv` +
  `validation_labels.csv` from the data folder). They are the only per-task truth, and the README
  allows using them. As a result, predictions on the validation week from `run.py` are *in-sample*.
  The honest numbers are the leave-one-day-out ones above.
- **The wait is the main remaining error.** Its service times are random, so a car that finishes
  near closing time may or may not slip to the next morning. Each wrong call costs 8–13 hours. On
  validation, 16 such cars account for 5.0 of the 8.9 min total MAE. The model predicts the median
  wait, which is the MAE-optimal choice, so most of this is luck. A Monte-Carlo queue simulation could
  sharpen the wait estimate near closing.
- **Repair is mostly noise.** Even with ~1,100 backed-out training examples, only reported damage
  (+~20 min) clearly explains it, so its MAE stays around 12–13 min.
- **Not done yet:** the README also asks for a 1-minute demo (`submission/demo.mp4`) or up to 5 slides
  (`submission/slides.pdf`).
