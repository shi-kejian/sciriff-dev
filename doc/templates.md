# Templates

- [Template schema](#template-schema)
- [Task details](#task-details)
- [Template details](#template-details)
- [Output formatting](#output-formatting)
- [Converting from Promptsource](#converting-from-promptsource)
- [Using datasets that aren't on Huggingface](#using-datasets-that-arent-on-huggingface)

## Template schema

To contribute a new task, create a YAML template for it. Templates go in the `tasks/templates` directory. Once you've written your template and confirmed that you can [instantiate](../README.md#instantiating-prompted-datasets-from-tasks) it into a dataset, create a PR.

Our template format is similar to [promptsource](https://github.com/bigscience-workshop/promptsource), with a few modifications. The format is below. We use `?` in the schema to indicate that a field is optional and can be ommitted.

Unlike promptsource, a single yaml file should correspond to a _single task_ associated with a dataset. Multiple prompts associated with this task are welcome.

```yaml
name: string              # Your name for this task.
dataset?: string          # If available, the Huggingface dataset used for the task. If null, load from file instead.
subset?: string           # Optionally, the subset of the Huggingface dataset.
data_dir?: string         # If not on Huggingface, the data directory, specified relative to `data/processed`.
skip_splits?: list        # Some HF datasets have blind test sets with blank outputs; these should be skipped.
split_lookup?: dict       # If needed, mapping from canonical split names to names for this HF dataset.
standardize_splits?: bool # True by default. If false, don't re-balance the data to create 3 splits.
formatter?:               # Optionally, a formatter used to preprocess the dataset.
    name: string
    args:
        key1: value1
        key2: value2
        ...
evaluate?:                       # Evaluation protocol for this dataset.
  metrics: list                  # The list of metrics to apply for this dataset.
    - metric1
    - metric2
    ...
  output_transform: string       # If specified, a function to apply to LLM outputs before running eval.
metadata:                        # Task metadata
  task: string                   # The task type. For example, extractive question answering.
  domains: list[string]          # A list of domains for this dataset.
  source_type: string            # single doc or multiple?
  input_context: string          # The input context for this task.
  output_context: string         # The output context (e.g. a label, a paragraph).
  contributor: string            # An ID for the contributor (you).
templates:                       # The templates for this task.
  0:                   # Everything is the same as promptsource templates, except we
    jinja: string                # number the templates starting at 0 rather than using unique ID's.
    evaluate?: bool              # Mark `true` if this template should be used for evaluation.
    answer_choices?: string
    name?: string
    metadata?:
      original_task?: bool
      choices_in_prompt?: bool
      metrics?: [string]
      description_loc?: string   # One new field: where does the task description go?
  1:                             # Keep counting.
    ...
```

## Task details

- `name`: The `name` field should match the name of the `yaml` file containing it. It should have the form `<dataset name>_<task name>`. For instance, if your source dataset is Qasper and the task is abstractive QA, a good `name` would be `qasper_abstractive_qa`.
- `dataset`: The name of the Huggingface dataset used to load the model. For Qasper, this would be `allenai/qasper`. For datasets loaded from a local directory, skip this.
- `subset`: Some Huggingface datasets have a `subset` field; some do not. Leave this blank for datasets with no `subset`.
- `data_dir`: For datasets loaded from a local directory, this should be the name of the directory located under `data/processed`.
- `skip_splits`: Some datasets (for instance MSLR 2022) have a test split available but its outputs are blank. List any splits here that should be skipped during instantiation.
- `split_lookup`: The standard Huggingface dataset split names are `train`, `validation`, and `test`. Some datasets use different names for some folds. For instance, [SciRepEval](https://huggingface.co/datasets/allenai/scirepeval) uses `evaluation` instead of `test`. This field is a dict mapping from the canonical split names to the names for this particular dataset, if needed. for instance, for SciRepEval, we could use:
  ```yaml
  train: train
  validation: validation
  test: eval
  ```
- `standardize_splits`: By default, if some splits are missing, the instantiation code will re-balance the data to create 3 splits. In some cases this may be undesirable; if so, just set `standardize_splits: false` and the code will preserve whatever splits are available on Huggingface.
- `formatter`: Sometimes, there's no way to conveniently write a Jinja template to directly format a dataset as it appears in Huggingface. When this occurs, specify a formatter. The formatter's job is to take a `datasets.Dataset` as input, and return a list of instances as output, such that each output instance has the fields expected by the Jinja templates for this task. Formatters go in `tasks/formatters`. The `name` of the formatter should match a filename in `tasks/formatters`, with the `.py` extension removed. The formatter script should define a class `Formatter` with a method `format_instances`; this is the method that will take a `datasets.Dataset` and return a list of formatted instances. For an example, see the [qasper formatter](../tasks/formatter/qasper.py).
  - The `args` to the formatter specify additional (optional) arguments that will be passed to the `Formatter` constructor. This can be useful if you want to use the same formatter for different tasks based on the same dataset.
- `evaluate`: Evaluation procedure for this task. Details on evaluation can be found in [evaluation.md](./evaluation.md).
  - `metrics`: Evaluation metrics to use.
  - `output_transform`: For some tasks (for instance, IE), it may be necessary to transform the LLM's output to a structured form before performing evaluation. Specify the transform to use.
- `metadata` for the task.
  - `tasks` The task should be one of the following:
  - `domain`: The domain should be one of the following:
  - `input_context`: How much document context is included in the input? The options here are:
    For lists of accepted values for tasks, domains, and input contexts, see [task_categories.yaml](../tasks/task_categories.yaml). If you think there's a category that's missing, feel free to add it in a PR.
  - `contributor`: An ID for the contributor (you). Please use either your github ID or an email address.

## Template details

Unlike in promptsource, templates are numbered with ints starting from 0; this is just simpler. Each template has the following fields (some optional):

- `jinja`: The jinja template. Exactly the same as promptsource; input and output are separated by `|||.` Since your Jinja template will likely be a multiline string, please use a [block scalar](https://yaml-multiline.info/) `|` to indicate a multiline string. For example:

  ```yaml
  ...
  <!-- Do this -->
  jinja: |
    For this task, please write a story about a miniature goldendoodle.

    The dog's name should be {{ name }}.

  <!-- Not this -->
  jinja: 'For this task, please write a story about a miniature goldendoodle.

    The dog's name should be {{ name }}.'
  ```

- `evaluate`: If `true`, use this template for evaluation on this task. At most one template for a given task should have `evaluate` set to `true`.
- `answer_choices`: Optionally, the list of answer choices, separated by `|||`. Exactly as in promptsource.
- `name`: Optionally, a name for the template (feel free to leave this blank).
- `metadata`: Metadata specific to this template. All fields are the same as promptsource, except:

  - `description_loc`: Where is the task description located in the prompt?

    - `before`: Task description comes before the input text. For example:

      ```text
      In this task, you will answer a question about a scientific paper

      Paper: [Paper text]

      Q: Who were the patients in the study?
      ```

    - `after`: Task description comes after input text.
      Paper: [Paper text]

      Your task is to answer a question about the paper shown above.

      Q: Who were the patients in the study?

    - `none`: No task description

      ```text
      Paper: [Paper text]

      Q: Who were the patients in the study?
      ```

    - `interleaved`: Task description is intervleaved with input text.

For guidelines on writing good templates, see [guidelines.md](guidelines.md).

## Output formatting

Many of the tasks we're working with have a structured output. Having a standardized output format for these will be very useful for evaluation. Here's the convention we'll use: for the first template associated with a given task with structured output, the model should output a `json` data structure. Also, make sure in the instruction to mention something like `please return the json object and no other text`. Here are suggested ways to format different types of structured outputs. Feel free to add to this list if more examples come up.

- Named entity recognition: `json` object where keys are entity names, and values are entity types. `{<ent1>: <type1>, <ent2>: <type2>, ...}`
- Binary relation extraction: `json` array where each entry is a relation triple. `[[<ent11>, <rel1>, <ent12>], [<ent21>, <rel2>, <ent22>], ...]`
- N-ary relation extraction: `json array` where each entry is a relation tuple and the relation comes at the end. `[[<ent11>, <ent12>, ..., <ent1N>, <rel1>], [<ent21>, <ent22>, ..., <ent2N>, <rel2>], ...]`.
- Attributed question answering or entailment: `json` object where one field is the answer and one is the evidence. `{"answer": <the_answer>, "evidence": [<evidence_1>, <evidence_2>, ...]}`.

## Converting from Promptsource

In summary, to convert from Promptsource-style templates to our template format, make the following changes:

- For the task information:
  - Add a `name` field.
  - Change the `dataset` and `subset` fields to specify the name of the huggingface dataset to format.
  - Leave the `formatter` field blank.
  - Add metadata on `task`, `domain`, and `input_context`.
- For each template:
  - Convert the keys to ints counting from 0.
  - If possible, add a `description_loc` field to the metadata.

## Using datasets that aren't on Huggingface

To format datasets that aren't available on Huggingface, you have the following options (in order of preference).

**1.** Add the dataset to Huggingface and proceed as usual.

**2.** Write a script to download and format your data. For an example of this, see [get_scierc.sh](../tasks/script/get_scierc.sh). The script should be run from the `tasks` folder; specify paths relative to this.

- Put any downloaded raw data in a subfolder of `data/downloads`.
- Do whatever processing you need and put the processed data in a subfolder of `data/processed`. The processed data should:
  - Consist of `.jsonl` files, with one line per instance.
  - Be named like `{train|validation|test}.jsonl`. It's OK if not all three folds are there, but there shouldn't be folds with other names.
- Once you're satisfied with your data, add your runner script to [tasks/script/get_all_datasets.sh](tasks/script/get_all_datasets.sh). That way, users can download all datasets with a single call.
- Now, you can create a template as usual. Leave the `dataset` field blank, and instead use the `data_dir` field to specify the folder containing your processed data. See the [scierc](task/templates/scierc_ner.yaml) template for an example.

**3.** In rare cases, there may be datasets that aren't even downloadable. In this case we'll use git lfs:

- Put the dataset in `tasks/data/lfs_data`
- `git add` your dataset and confirm that it's being tracked by lfs with `git lfs ls-files`.
- Commit any relevant files and include them in your PR.
- Add information on your datasets to [lfs_datasets.md](./lfs_datasets.md) so it's clear where they came from.
- Proceeed as before, writing a preprocessing script and putting the processed data in `tasks/data/processed`. If there's no preprocessing necessary, just write a script to copy the files over.

**4.** Some datasets are downloadable but are behind paywalls or only available after registering for a website (BioASQ is an example). If you have a dataset like this, add the data download script to [get_protected_datasets.sh](tasks/script/get_protected_datasets.sh) rather than [get_all_datasets.sh](tasks/script/get_protected_datasets.sh).
