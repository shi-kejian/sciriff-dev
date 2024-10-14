# Science adaptation

Repo for LLM adaptation for scientific tasks. Goals for this repo include:

- Formatting a variety of existing scientific tasks for instruction tuning.
- Training models and performing evaluations on these existing tasks.
- Collecting data and performing evaluations for novel scientific tasks.

**Table of contents**

- [Setup](#setup)
- [Getting the data](#getting-the-data)
- [Building the dataset](#building-the-dataset)
- [Running evaluation](#running-evaluation)
- [Contributing tasks](#contributing-tasks)

## Setup

```bash
conda create --name science-adapt python=3.11
conda activate science-adapt
conda install conda-build

# At some point we'll do `pip install -e .` instead of `conda develop`; need to re-organize the repo first.
python setup.py
conda develop .

# Set the root of the project on your filesystem
conda env config vars set PROJECT_ROOT=[path_to_this_repo]
```

You will also need to have git LFS installed; see [here](https://git-lfs.com/) on how to do this.

Finally, if you want to recreate the dataset, run the commands below. If you just want to work with the existing data, you can ignore this.

```bash
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

## Getting the data

The full dataset including all task instances is available here: <https://huggingface.co/datasets/ai2-adapt-dev/science-adapt-4096>. `4096` refers to the context window; there are also `8192` and `16384` versions but we haven't experimented with those. If you look at the [files and versions](https://huggingface.co/datasets/ai2-adapt-dev/science-adapt-4096/tree/main), you'll see a sub-directory for each task. Each task has `train`, `validation`, and `test` splits.

To load a single task, you can do, for example:

```python
import datasets

ds = datasets.load_dataset("ai2-adapt-dev/science-adapt-4096", "qasper_abstractive_qa")
```

TODO also put the training mixtures up.

## Building the dataset

If you want to build the data from scratch, you can do that. There are two steps: first, we need to create task instances from their [templates](tasks/templates). Then, we need to mix the instances together, and potentially combine with Tulu data.

### Creating task instances

Run [build_dataset.sh](tasks/build_dataset.sh). This performs the following steps:

- Validates all templates to make sure they match the [task schema](tasks/task_schema.yaml).
- Downloads data needed for tasks that aren't available on Huggingface.
- [instantiate](tasks/instantiate.py)s all tasks and puts them in `data/instances`. This dumps a max of 2,500 instances / task, which is more than we need for instruction tuning.

Within AI2, instead of running these yourself you can just use the data at `/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/instances`; I'll try to put this on Beaker at some point.

The `instances` directory is organized as follow:

```bash
| context_window
  | task_name
```

I've created versions of the data with context windows of 4,096, 8,192, and 16,382. I've only ran experiments on 4,096 so far; this is the max that you can train on A100's.

If you want to look at one of the tasks in a "pretty-printed" format, use [preview.py](tasks/preview.py); for instance, from the root of the project, you can run

```bash
python tasks/preview.py \
  --instance_file data/instances/4096/scifact_entailment/train.jsonl \
  --n_instances 10 \
  --format_json
```

### Creating training mixtures

You can create training mixtures using [create_mixture.py](mixture/create_mixture.py). This puts all the science data in tulu-friendly format; you can point open-instruct at the resulting mixture file and start training. I've kept the options very simple. See the command line flags for full info. The important options are:

- `instances_per_task`: How many instances per science task? I've found that performance saturates at about 1000; it's worth experimenting with less.
- `tulu`: How much tulu data to mix in? Current options are:
  - `all`: Use it all; appropriate for finetuning from a pretrained model checkpoint. Training on a mix with all tulu takes roughly 8 A100-days.
  - `match`: Match the number of science instances; appropriate for continued finetuning from an instruction-tuned model checkpoint. Continued finetuning will take about 8 A100-hours.
  - `none`: Don't use any; can also be used for continued finetuning, but performance on open-instruct tasks will take a hit.
- `include_eval_tasks`: Whether to include the eval tasks during training. The eval tasks are the yaml files included in [eval/tasks/tulu](eval/tasks/tulu), excluding `_default_template.yaml`. I'm going to clean up the evals but you can go on these for now.

Within AI2, I've already created a bunch of training mixtures using [create_all_mixtures.py](mixture/create_all_mixtures.py). The results live at `/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/training_mixtures`. These are organized into directories by context window. A single mixture file is named like `tulu_{how much tulu}_science_{science insts per task}_{eval}_{yes/no}.jsonl`.

If rebuilding yourself, you can run [test_mixtures.py](mixture/test_mixtures.py) to do some quick checks to make sure things look OK.

## Running evaluation

Evaluation happens in two stages. First, we use the Eleuther harness to make predictions for the eval tasks. Then, we compute metrics on top of the predictions (I found this was more flexible than using the Eleuther harness for the full eval pipeline). Eval code in progress, but you can get things going like so:

For more details on how the evals are implemented (useful if you want to extend or modify), see [evaluation.md](doc/evaluation.md).

### Making predictions

Make predictions using [predict_eleuther.py](eval/script/predict_eleuther.py). You can make predictions for API models like GPT-4, or models that you've trained.

For external use (non-AI2), pass the `--interactive` flag to run evaluation directly in your current interactive environment. For internal AI2 use, it's easier to kick off a batch evaluation job on Beaker. Call `predict_eleuther.py -h` for more information on all the available arguments. All arguments that start with `beaker_` are only relevant for Beaker and will be ignored otherwise.

Here's an example call to run evaluation on a Huggingface tulu model in an interactive environment with a GPU:

```bash
python predict_eleuther.py \
  --model vllm \
  --model_name allenai/tulu-2-7b \
  --result_base [path-for-eval_results] \
  --tulu_format \    # Pass this flag for models trained on Tulu-style instruction data.
  --gpus 1 \
  --interactive
```

This script will create a directory titled `tulu-2-7b` underneath `result_base` and place all predictions in that directory. It will make predictions for all eval tasks, listed here [here](eval/eleuther_templates/tulu/) (one yaml file per task); each task will get its own directory. Each task directory will have two files:

- `eleuther.jsonl` is the "metrics" output by Eleuther; this can mostly be ignored.
- `(pretrained|model)__{model_name}_{task_name}.jsonl` will have a dump of the predictions for each instance.

Eleuther doesn't have a flag specifying whether to evaluate on test or validation, so by default I point it toward the validation set (see [here](eval/eleuther_templates/tulu/_default_template.yaml)). You can edit this if you want to evaluate on test instead; I'll try to add a flag at some point.

If you instead want to evaluate a model checkpoint stored on disk, you can do

```bash
python predict_eleuther.py \
  --model vllm \
  --model_name [path-to-my-model] \
  ...
```

If you're AI2-internal and want to kick off a Beaker job to run evals, you can do:

```bash
python predict_eleuther.py \
  --model vllm \
  --model_name [path-to-my-model] \
  --result_base [path-for-eval_results] \
  --tulu_format \    # Pass this flag for models trained on Tulu-style instruction data.
  --gpus 1 \
  --beaker_cluster ai2/s2-cirrascale-l40 \
  --beaker_budget ai2/oe-adapt \
  --beaker_dataset [dataset-id] \ # Dataset with model checkpoint; will be mounted at `path-to-my-model`.
```

### Computing metrics

Computing metrics on top of the predictions is very quick, and is done with [compute_science_metrics.py](eval/script/compute_science_metrics.py).

Point to the script at the directory containing predictions for all your models (this should be the same as the `result_base` arg from the prediction step.), and give it an output directory to dump the metrics. You should also specify a `baseline_model`, which is the directory name of one of the models in `pred_dir`; this model will be used as a baseline to compare other models against for LLM-as-a-judge-style evaluations (currently just MUP).

```bash
python compute_science_metrics.py \
  --pred_dir [model_predictions] \
  --metrics_dir [output_metric_files] \
  --baseline_model [name-of-baseline-model]
```

The script will find all available predictions and compute metrics for them. The `metrics_dir` will be populated with:

- A `by_model` directory which has the metrics organized by model and then by task.
- A `tables` directory which collects all the results into tables. `summary.tsv` has summary metrics over all the tasks and compute a single task aggregate `mean` and `median`; the rest of the files are detailed results for individual tasks.

This is a work in progress; some of the metrics (particularly for evidence inference and Qasper) are still pretty noisy and don't provide much signal. Will try to fix this soon.

## Contributing tasks

Instructions on how to contribute a new task are found in [templates.md](doc/templates.md). Prompt-writing guidelines are in [guidelines.md](doc/guidelines.md). The task contribution process uses a similar templating approach to [promptsource](https://github.com/bigscience-workshop/promptsource), and requires additional metadata on the task category and domain a la [natural instructions](https://github.com/allenai/natural-instructions).

For more information on converting promptsource templates to our format, see [here](doc/templates.md#converting-from-promptsource).

Once you've added your templates, you can submit a PR for review. Before submitting your PR, please perform the following:

- Make sure to run [instantiate.py](tasks/instantiate.py) (see below) for each template to confirm that it runs without error and the outputs are as expected.
- Run [validate.py](https://github.com/allenai/science-adapt/blob/main/tasks/validate.py) and confirm it doesn't throw any errors.
- Run [black](https://github.com/psf/black) on any Python code you contribute (TODO: automate this).

### Tasks to contribute

We're tracking tasks in this [spreadsheet](https://docs.google.com/spreadsheets/d/1aOaaMpjLZoKt-It6tFdn9Vac8lkeOORnsSy1QlbNvYI/edit?usp=sharing). If you see a task whose status is `Needs to be added`, or `Needs review`, feel free to add it.

You're also welcome to contribute your own tasks. Feel free to include any tasks related to scientific literature understanding -- in other words, the input should include some piece of scientific literature. Tasks that are generally "about" science but don't work with scientific literature are out-of-scope for now.
