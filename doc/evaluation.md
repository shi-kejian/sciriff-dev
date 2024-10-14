# Evaluation

This doc has more details on how the evaluation code is organized. For high-level information see, the main [README](../README.md)

## Making predictions

The [predict_eleuther.py](../eval/script/predict_eleuther.py) uses Eleuther to run predictions. The way Eleuther works for prediction is: point the evaluator at a directory, and give it a template specifying how to format each instance. The templates for Tulu vs. other language models are slightly different; Tulu expects the input to be formatted like `<|user|>\n{{ input }}\n<|assistant|>`.

The configs to specify how evaluation works are in [eleuther_templates](../eval/eleuther_templates/), with one directory for `tulu` and one for `general`. You shouldn't have to mess with these unless you think that poor performance is due to something about the way the prompt is formatted, in which case you might want to modify the prompt template and re-run prediction.

## Evaluating

The evaluation script [compute_science_metrics.py](../eval/script/compute_science_metrics.py) takes a directory with a collection of model predictions as input and computes metrics for all the predictions.

**AI2-internal eval**: There are now a couple tasks that have "gpt-judge" evaluations; it's quicker to parallelize these rather than running sequentially for each model. To do this, run [launch_compute_science_metrics.sh](../eval/script/launch_compute_science_metrics.sh) to compute evals for all models in parallel, and then call [compute_science_metrics.py](../eval/script/compute_science_metrics.py) to aggregate all the results into a summary table..

### Eval implementation

The evaluations for each task are implemented in [eval/tasks](../eval/tasks/). Roughly, there are two types of tasks: those that expect a `json` output and those that don't. The ones with the `json` require some extra handling to try to extract the `json` output, keep track of how many parse failures there were, supply a default value in case of failures, and compute metrics over (1) all the instances, replacing failures with default values, and (2) just the instances that parsed successfully. This is all handled in the `JSONTask` class in [_base.py](../eval/tasks/_base.py).

Individual task evaluations inherit from one of the classes in `_base.py`. For instance, [biored_ner](../eval/tasks/biored_ner.py) inherits from `JSONTask` and runs json parsing before doing anything else.

After the initial preprocessing is done, each `task` calls out to a metric (or metrics) to compute results. The metrics are implemented in [eval/metrics](../eval/metrics/). There's one file per type of metric; for instance, [ner_f1](../eval/metrics/ner_f1.py) computes F1 scores for NER tasks. This metric is consumed by the [biored_ner](../eval/tasks/biored_ner.py) task. After the metrics have been computed, `biored_ner` collects allthe results together and adds some extra information -- for instance, on the `json` parse success rate -- and dumps results to `metrics.json` in the `biored_ner` subfolder of the model prediction directory.

Once all metrics have been computed and stored, the evaluation script collects them all and puts them in spreadsheets for easier inspection.

## Evaluation metrics

Detailed metrics on a few difficult-to-evaluate tasks are below.

In general:

- `json_parsed` is the fraction of cases where we could parse the string returned by the model as `json`.
  - `valid_json_*` is all metrics computed over the cases where we could parse the return value.
  - `all_*f` is all metrics computed over all cases, treating failures as `""`.

### Qasper

- `*answer_frac_parsed`: Fraction of time we could get an answer field out of the json.
- `*evidence_frac_parsed`: Fraction of time we could get evidence out of the json.
- `*is_str_frac_parsed`: Fraction of time the answer was a string (as opposed to a list or dict).

### Evidence inference

- `*frac_valid_list`: Fraction of the time the `json` was formatted as a list of 5-tuples, as required for evidence inference.
