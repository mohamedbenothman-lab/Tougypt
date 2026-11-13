# Branch 5.5 validation experiments

Run from the checkout root:

    python -B starter/validate_cv.py
    python -B starter/validate_cv.py --forward

The script compares a direct elapsed model and a next-opening carryover
classifier on identical rows. It uses the supplied weekly schedule, including
the next opening after a weekend closure. The task experiment compares pooled
medians with task-specific trees and a staffing-level tyre-check median.
Validation labels used to fit durations exclude the scored day. Forward mode
uses strictly earlier days and skips days without a fitting example for every
task. The reported elapsed comparison is against the direct model in this
script. The original 8.9-minute implementation was absent from this checkout.

Leave-one-day-out (2,791 rows): elapsed MAE 16.340 to 12.300 minutes,
last-hour MAE 116.198 to 73.290 minutes, mean task MAE 2.861 minutes,
and task WAPE 7.93%. Among the 16 largest last-hour misses, seven missed
a carryover and four predicted a carryover that did not occur.

Forward-day (2,477 rows): elapsed MAE 17.469 to 12.414 minutes,
last-hour MAE 123.625 to 68.844 minutes, mean task MAE 3.621 minutes,
and task WAPE 8.74%.

Per-row results and the 16 largest last-hour errors are saved in starter/
with the evaluation mode in the filename. These are diagnostic predictions
with actual outcomes and baseline columns included.

The report's inferred repair labels and original queue/work simulator were
unavailable, so these runs do not compare with and without those repair labels
or reproduce the original model. Preserve the reported 8.9-minute model as the
team benchmark once its source and exact validation predictions are available.

## Same-shift duration model

The candidate now fits a separate elapsed-time regressor on near-closing
training returns that were ready before the next scheduled opening. The
carryover decision and post-opening model are shared with the first experiment.

Leave-one-day-out elapsed MAE improved from 12.300 to 11.320 minutes, and
last-hour MAE improved from 73.290 to 61.034 minutes. Forward-day elapsed MAE
improved from 12.414 to 10.934 minutes, and last-hour MAE improved from 68.844
to 50.525 minutes. Task scores stayed the same. The CSVs reflect this run.

## Closing-time features

Added task flags, branch indicators, weekday, time of day, and estimated
minutes of slack before closing. At the original 0.50 decision threshold,
elapsed MAE was 9.875 minutes forward and 10.202 leave-one-day-out.
Forward last-hour MAE rose from 50.525 to 57.635 while earlier-open MAE
fell from 7.990 to 6.329. Task scores were unchanged. A business-clock
alternative scored 11.302 forward. Threshold sweeps selected different
values by protocol, so the original threshold was retained. Both scores
are public development results after model selection.
