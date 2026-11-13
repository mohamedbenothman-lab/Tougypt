# Task Time Detective: submission

## Results (validation week, leave-one-day-out, so no model saw the day it predicts)

| Metric | Baseline (`solution.py` as given) | This model | Estimated ceiling | README "Strong" |
|---|---|---|---|---|
| Per-task WAPE | ~0.55 | **0.0746** | ~0.072 | < 0.2 |
| Mean per-task MAE (all 9 tasks weighted equally) | – | **2.67 min** | ~2.57 min | – |
| Total turnaround MAE | ~99 min | **7.69 min** (5.82 on held-out train weeks) | ~5.5–7.5 min (random queue stays; see below) | < 5 min |
| Negative or impossible durations | – | 0 | – | – |

- The total is in the "Good" band (< 15 min). Most of it comes from **13 of 2,791 cars** whose work
  ends right around closing time, where a few minutes of queue decide between "ready tonight" and
  "ready tomorrow morning". Without those cars, the total MAE is 3.5 min.
- Two runs give byte-identical output (`run.py --repeat`, plus a separate process run). Fit + predict
  takes about 5 s on a laptop.
- On the same rows and metrics, the `codex-sixt` branch's model scores 2.86 mean per-task MAE and
  10.20 total MAE.

## How the turnaround time is built

Checked against the true per-task times in `validation_labels.csv`:

1. **Each car's tasks follow a fixed priority schedule across the staff.** The total equals the sum of
   task times only 5% of the time, so tasks run in parallel across the branch's `staff_on_duty`
   workers. The exact rule was found by checking cars that arrived at a quiet branch, where the total is
   pure work time:
   1. A **quick damage repair comes first**: damage check, then the repair, with the car alone. Every
      other task waits for it.
   2. The remaining tasks are handed out in a **fixed order** (exterior → special cleaning → tyre check →
      interior → damage check → refuel), each to whichever worker is free first.
   3. **Interior cleaning waits for special cleaning**, since both happen inside the car.
   4. **Electric charging runs on the charger in parallel** and needs no staff.

   This reproduces **97–98% of quiet-branch totals exactly**, against 82% for "longest task first".
   With 1 staff (city centers) everything is sequential, so the rule only matters at stations and
   airports. Between cars the queue is first come, first served. Together, that is the "mix of
   priority and FIFO".
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
4. **Task durations are additive.** For example, a child seat adds the same ~7.6 min of interior
   cleaning at every dirt level. A median linear model therefore fits them with far fewer parameters
   than trees, and beat gradient boosting on every frequent task.

## Approach (`starter/solution.py`)

| Part | Model | Trained on |
|---|---|---|
| 7 frequent tasks | additive **median (L1) linear regression** per task, on standardised columns, L1 penalty chosen per task | validation labels (true task minutes) |
| `tyre_check` | median per branch staffing level (1/2/3 staff) | validation labels (29 rows; any fitted model overfits) |
| `quick_damage_repair` | small gradient-boosted trees | ~1,100 durations backed out of `train.csv` totals + the 71 validation labels |
| waiting time | gradient boosting, absolute-error loss | `train.csv` totals plus the validation week's totals: open minutes until ready minus the car's predicted makespan |
| total | `return + open-minute clock(predicted makespan + predicted wait)` | assembled, no extra fitting |

- **Task inputs:**
  - Car condition: dirt scores as levels, child seat, pet, smoking, damage.
  - Car and rental: vehicle class, powertrain, fuel/battery level, km, rental days.
  - Context: branch, weather, hour, weekday, `queue_length`, and the other task flags.

  Predictions are clipped to the observed range, and a task that was not done gets 0.
- **L1 penalty:** 0.001–0.003 for the cleaning tasks and refuel, 0.01 for special cleaning and the damage
  check, 0.04 for charging. With these settings the damage check keeps only `damage_reported`, and
  charging keeps only the battery level.
- **Repair from totals:** only 71 repairs are labelled, but `train.csv` has ~1,100. For each one, the
  repair time is the value r that makes `max(staff schedule with repair = r, charging)` equal the
  observed open minutes minus the predicted wait (solved by bisection). On validation, where the
  truth is known, these backed-out values are within ~4.2 min of the true repair time (median bias
  −1 min).
- **Wait inputs:**
  - Arrivals at the same branch in the previous 10–180 open minutes, and cars waiting from the same
    closure (key-box backlog).
  - Minutes since opening and until closing, the slack left before closing, and how long the branch
    then stays closed.
  - Small deterministic queue simulations, staff, branch, hour, weekday, and the car's own makespan.

  Only past arrivals are used, never future ones.
- **Determinism:** fixed `random_state`, no early stopping, no sampling. The linear models are
  solved exactly (HiGHS). `python run.py --repeat` checks that two runs are identical.

## How close is this to the ceiling?

**Per task.** Each figure below is bracketed by the honest leave-one-day-out MAE and a learning-curve
extrapolation to unlimited labelled data (MAE ≈ floor + c/n, using 50% and 100% of the labels):

| Task | Now | Estimated floor | Gap |
|---|---|---|---|
| quick exterior cleaning | 0.803 | ~0.78 | 0.02 |
| normal exterior cleaning | 1.610 | ~1.54 | 0.07 |
| special cleaning | 0.796 | ~0.80 | – |
| interior cleaning | 1.237 | ~1.23 | 0.01 |
| damage check | 0.196 | ~0.195 | – |
| refuel | 0.856 | ~0.78 | 0.08 |
| electric charging | 1.630 | ~1.62 | 0.01 |
| quick damage repair | 12.62 | ~12.2 | 0.4 (trees, linear, medians all land at 12.2–12.9: it is noise) |
| tyre check | 4.25 | ~4 | noisy, only 29 labels |

- **What's left:** boosting the linear models' residuals finds nothing, so no non-additive structure
  remains. The remaining gaps (normal exterior, refuel) come from estimating ~45 coefficients from
  410–1,216 rows. Only more labelled data would close them.

**Total.** The wait model was selected on a larger yardstick: fitted on train weeks 1–9 and scored on
train weeks 10–12 (~8,500 cars).

- **Achieved:** total MAE **5.82** there: 6.01 with the old "longest task first" schedule, and 3.52
  on the cars that don't flip across a closure.
- **Estimated floor:** quantile models of the wait give each car's full predictive spread. With the
  old schedule, the expected |error| of the median under that spread was **6.78** against an achieved
  6.01, so the wait model sits at the floor these inputs allow. The schedule fix lowered the floor
  itself by removing work-time errors that had looked like queue noise.
- **Alternatives tried:** more trees, deeper trees, extra queue simulations driven by each car's own
  predicted workload, and dropping features. None beat it beyond noise (bootstrap 95% CIs include 0).
- **The queue itself is random.** The true wait sits on a grid of n × 12 min ÷ staff, so the
  number of cars ahead (n) can be read off for every validation car.
  - n never grows by more than +1 per arrival, and it falls faster when more cars are present. Each
    car stays in the queue for its own random time, and cars left in the key box all queue at opening.
  - No departure rule tied to observable quantities reproduces n: not the car's own work, not its
    total, not a fixed time. The best gets 53% exact.
  - At the 1-staff centers, whether the previous car is still queued is predicted by the time gap
    alone (AUC 0.83). Adding any of that car's features (task times, makespan, dirt, hour, km,
    `queue_length`) makes cross-validated predictions worse. So the stay times are unobservable noise.
  - A model of the queue built on exactly that random-stay structure puts the best possible
    center wait error at ~6.0 min. The trained wait model scores 6.0 on the same cars.
- **It is not a priority queue.** Two priority rules were tested against FIFO: more jobs first, and
  less work first.
  - In the key-box backlogs, where cars join at the same moment, service order follows return time
    (Kendall τ 0.99) and ignores job count (τ 0.03).
  - Holding a car's own work fixed, earlier cars with less work and earlier cars with more work add the
    same number of cars ahead. All 95% CIs include 0, except airports, where heavier cars count
    slightly *more*.
  - "Heavier cars first" was also tested at stations alone, as a queue simulation and as an input to
    the wait model. It matched the real waits no better than first come first served (Spearman ρ
    0.422 vs 0.428). In the wait model it did not lower station error either: 5.73 → 5.75 min on the
    held-out train weeks.
  - With the old schedule, a car's **own** tasks seemed to change its place in the queue at 2–3 staff
    branches. Measured in cars ahead: special cleaning +3.0, charging +2.0, refuel −1.5, and about
    +0.3 min of wait per minute of own work. All of it was the within-car schedule being wrong: with
    the fixed-order schedule above, every one of these effects disappears (own work +0.000/min at
    airports).
  - The wait error drops at stations from 6.1 to 3.7 min and at airports from 6.7 to 4.3 min.
  - What is left between cars is first come, first served with random stays. Cars arriving later add
    nothing.
- **The labelled week's totals were also added to the wait model's training data.** It's the period
  closest to the test weeks. In leave-one-day-out the effect is within noise: total MAE +0.13 min
  (95% CI −0.29 to +0.79), repair MAE −0.17 min (CI −0.44 to +0.11). It was kept for its recency.
- **Where the floor comes from:** about 2% of cars could plausibly finish either before closing or
  after the next opening. Each wrong call costs from 6 hours (airport night) to a whole weekend (city centers),
  so a handful of cars sets the total MAE.
- **Week-to-week swings:** the validation week has more such cars (5 per 1,000 vs 3.2), which is why
  its total MAE (7.7) is higher than on the held-out train weeks (5.8).

## Effects learned (final models, in minutes)

| Task | Main effects | Does it make sense? |
|---|---|---|
| quick exterior cleaning | exterior dirt 3/4/5: +0.9/+1.3/+2.1; airports +0.9 to +1.2; +0.14 per car in the queue | yes: dirtier cars and busy airports take longer |
| normal exterior cleaning | exterior dirt 2/3/4/5: +1.8/+3.6/+5.4/+7.4; van +2.5, SUV +1.1; rain +1.6 | yes |
| special cleaning | smoking +7.9; pet +2.0; child seat +1.6; premium +2.7 | yes |
| interior cleaning | child seat +7.6; interior dirt 2/3/4/5: +2.2/+4.4/+6.6/+8.5; van +1.8; +0.14 per rental day | yes |
| damage check | only `damage_reported`: +0.3 (otherwise ~2.4 min) | yes |
| refuel | FRA +6.1, MUC +5.3, DUS +4.1, NUE −3.8 vs BER; van +4.1, SUV +2.9, premium +2.0 | plausible: distance to the pump by branch, tank size by class |
| electric charging | only battery level: −0.34 min per % (≈ 40 min when empty) | yes |
| quick damage repair | damage reported ≈ +20 min; the rest is mostly noise | yes |
| tyre check | median 47.0 / 33.3 / 17.5 min at 1 / 2 / 3-staff branches | plausible: small branches lack equipment and hands |

**Waiting-time model** (permutation importance: MAE increase in minutes when a feature is shuffled,
6,000 training rows): return_hour 5.33, ms_staff 3.60, min_since_open 3.17, min_to_close 2.47,
prev30 1.60, prev45 1.59, same_open_ahead (key-box backlog) 1.36, sim_wait_s20 1.18. The wait is
driven by time of day, how close closing is, and how many cars arrived just before. `queue_length` adds
nothing.

## Per-task estimates (validation week, leave-one-day-out predictions)

| Task | Cars (val) | True mean (min) | Predicted mean | MAE | MAE if always guessing the average | Skill | Typical estimate |
|---|---|---|---|---|---|---|---|
| quick exterior cleaning | 1,575 | 13.3 | 13.2 | 0.80 | 1.02 | 0.21 | 12.6 (clean) → 15.0 (dirt 5) |
| normal exterior cleaning | 1,216 | 19.2 | 19.1 | 1.61 | 2.29 | 0.30 | 16.3 (clean) → 23.5 (dirt 5) |
| special cleaning | 277 | 30.0 | 30.1 | 0.80 | 2.11 | 0.62 | ~29; ~37 if smoking reported |
| interior cleaning | 2,791 | 16.4 | 16.4 | 1.24 | 2.96 | 0.58 | 13.3 (clean) → 22.8 (dirt 5); +7.6 with a child seat |
| damage check | 1,394 | 2.4 | 2.4 | 0.20 | 0.21 | 0.08 | ~2.4; ~2.7 with reported damage |
| refuel | 410 | 13.5 | 13.4 | 0.86 | 2.86 | 0.70 | 7–17 depending on branch; larger cars longer |
| electric charging | 526 | 24.8 | 24.8 | 1.63 | 6.16 | 0.74 | ≈ 40 − 0.34 × battery % |
| quick damage repair | 71 | 68.6 | 66.6 | 12.62 | 15.12 | 0.17 | ~55 without reported damage, ~76 with |
| tyre check | 29 | 26.1 | 25.7 | 4.25 | 10.28 | 0.59 | 17.5 / 33 / 47 at 3- / 2- / 1-staff branches |

Skill = 1 − MAE / MAE of always guessing the average. As the README warns, it is harsh for tasks
that barely vary, such as the damage check.

## How to run

```bash
pip install -r requirements.txt        # Python 3.11+
cd starter
# submission file (data/test.csv must be in place), then the reproduction check
mkdir -p ../submission && python run.py --test ../data/test.csv --out ../submission/test_predictions.csv
python run.py --compare                # re-runs on ../data/test.csv; exits 1 unless it equals ../submission/test_predictions.csv
# validation
python run.py --repeat                 # fit on train + labels, predict validation.csv -> predictions.csv, check determinism
python validate_cv.py                  # honest leave-one-day-out scores -> predictions_cv.csv
python evaluate_local.py predictions_cv.csv --labels ../data/validation_labels.csv --flags ../data/validation.csv
```

The output has one row per turnaround: `turnaround_id`, `total_turnaround_min`, and `dur_<task>` for
all 9 tasks (0 when the task was not done). `submission/test_predictions.csv` was made with the command
above, and `run.py --compare` reproduces it with 0 differing values (8,354 rows × 10 values).
Fit + predict takes about 5 s.

## Caveats

- **Data location:** all data is in `data/`. `data/test.csv` is not committed (see `.gitignore`); put
  the released test file there before running the submission command.
- **Missing organiser files:** the official `evaluate.py`, `run.py` template and `setup.sh` are not
  in this repo. `run.py` follows the submission instructions (`--test`, `--out`, `--compare`), and
  `evaluate_local.py` re-implements the README's metrics.
- **`fit(train, branches)` also reads the validation labels** (`validation.csv` +
  `validation_labels.csv` from the data folder). They are the only per-task truth, and the README
  allows using them. As a result, predictions on the validation week from `run.py` are *in-sample*.
  The honest numbers are the leave-one-day-out ones above.
- **Model choices used the validation week:** the per-task model types and L1 penalties were picked by
  leave-one-day-out MAE on it, from small grids. The wait model was picked on held-out train weeks
  instead.
- **Slides:** `submission/slides.pdf` (5 slides) summarises the approach and results.
