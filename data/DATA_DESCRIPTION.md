# Data description

`train.csv` (with `total_turnaround_min`), `validation.csv` (without it) and `validation_labels.csv` (the total and the true
duration of each task for the validation turnarounds). One row per returned vehicle. Validation comes after the training
period; the hidden test set comes after validation.

| Column | Meaning |
|---|---|
| `turnaround_id` | unique id |
| `return_ts` | time the vehicle came back; while a branch is closed, vehicles can be left in the key drop box |
| `branch`, `vehicle_class`, `powertrain` | where, what kind of car (`petrol_diesel`, `hybrid`, `bev`) |
| `rental_days`, `km_driven` | rental length and distance |
| `return_hour`, `weekday` | derived from `return_ts` (0 = Monday) |
| `fuel_level_pct`, `battery_soc_pct` | level at return; blank when not applicable to the powertrain |
| `interior_dirt`, `exterior_dirt` | condition score 1 (clean) to 5 (very dirty) |
| `child_seat`, `pet`, `smoking_reported`, `damage_reported` | flags from check-in |
| `weather` | `dry`, `rain` or `snow` at return |
| `queue_length` | cars waiting at the branch at return |
| 9 task columns | 1 if the task was performed on this turnaround, else 0 |
| `total_turnaround_min` | minutes from `return_ts` until the vehicle was ready (train only; in `validation_labels.csv` for validation) |

`validation_labels.csv` also has one column per task with its true duration in minutes (0 if not performed).
`branch_data.csv`: opening and closing time per branch and weekday, and `staff_on_duty` per branch.

Tasks: quick_exterior_cleaning, normal_exterior_cleaning, special_cleaning, interior_cleaning, damage_check, refuel,
electric_charging, quick_damage_repair, tyre_check. The data is synthetic; no real customer or company data.
