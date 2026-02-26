# Arena Human-vs-Judge Consistency Report

## Dataset
- File: arena_math_900.json
- Rows loaded: 894
- Rows used for comparison (both human and judge labels present): 894/894 (100.0%)
- Rows excluded from comparison: 0
- Human label distribution (all rows): model_a=290 (32.4%), model_b=263 (29.4%), tie=190 (21.3%), both_bad=151 (16.9%)
- Judge label distribution (all rows): model_a=312 (34.9%), model_b=278 (31.1%), tie=304 (34.0%)
- Judge coverage: 894/894 (100.0%)
- Human label distribution (comparison subset): model_a=290 (32.4%), model_b=263 (29.4%), tie=190 (21.3%), both_bad=151 (16.9%)
- Judge label distribution (comparison subset): model_a=312 (34.9%), model_b=278 (31.1%), tie=304 (34.0%)

## Agreement Metrics
- Legacy decisive agreement (human/model_a|model_b vs judge/model_a|model_b): 253/381 (66.4%)
- Full-label agreement (all non-missing human and judge labels): 347/894 (38.8%)
- Decisive judge coverage on human-decisive items: 381/553 (68.9%)

## Typical Forms of Inconsistency (Human vs Judge)
- Total disagreements on comparison subset: 547/894 (61.2%)
- decisive_flip_a_vs_b: 128 (23.4% of comparison disagreements)
- decisive_vs_tie_or_both_bad: 381 (69.7% of comparison disagreements)
- tie_vs_both_bad: 38 (6.9% of comparison disagreements)
- other_labeled_mismatch: 0 (0.0% of comparison disagreements)
- Top labeled mismatches (human->judge): model_a->tie: 90, model_b->tie: 82, model_b->model_a: 66, both_bad->model_a: 65, model_a->model_b: 62, tie->model_b: 53, both_bad->model_b: 48, tie->model_a: 43

## Confusion Matrix
- 4x4 layout: rows are human labels, columns are judge labels.
- Label order: model_a, model_b, tie, both_bad.

## Interpretation Note
- With one judge source and no external gold signal, this report quantifies disagreement patterns only; it does not prove whether human or judge labels are universally better.
