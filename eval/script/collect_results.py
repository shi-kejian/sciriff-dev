"""
Collect results from science and open instruct evals.
"""

# This code should probably be cleaned up, but it works OK.


import pandas as pd
from functools import reduce, partial
import json
from pathlib import Path

from lib import paths


########################################


def put_baselines_last(df):
    """
    Put baseline results at end of table.
    """
    baselines = [
        "Llama-2-7b-chat-hf",
        "tulu-2-7b",
        "tulu-2-dpo-7b",
        "Llama-2-70b-chat-hf",
        "tulu-2-dpo-70b",
        "tulu_2_70b_no_science",
        "claude-2",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "tulu-2-13b",
    ]
    baseline_rows = df.loc[baselines].index
    remaining = df.index.difference(baseline_rows)
    new_order = list(remaining) + list(baseline_rows)
    return df.loc[new_order]


########################################

# Science


def get_nested_value(nested_dict, keys_tuple):
    return reduce(lambda d, key: d[key], keys_tuple, nested_dict)


def get_science_results():
    # Mappings from the metrics dictionary structure to the outputs we want.
    lookup = [
        [("biored_ner", "bleu,extract-json"), ("biored_ner", "bleu")],
        [("biored_ner", "rouge,extract-json", "rougeL"), ("biored_ner", "rougeL")],
        [
            ("biored_ner", "f1_ner_exact,extract-json", "f1_typed"),
            ("biored_ner", "f1_ner"),
        ],
        [("discomat_te", "bleu,none"), ("discomat_te", "bleu")],
        [("discomat_te", "rouge,none", "rougeL"), ("discomat_te", "rougeL")],
        [("evidence_inference", "bleu,extract-json"), ("evidence_inference", "bleu")],
        [
            ("evidence_inference", "rouge,extract-json", "rougeL"),
            ("evidence_inference", "rougeL"),
        ],
        [
            ("evidence_inference", "f1_relation_exact,extract-json", "f1"),
            ("evidence_inference", "f1_rel"),
        ],
        [
            ("multicite_intent_classification", "f1_list_exact,extract-json", "f1"),
            ("multicite_intent_classification", "f1_label"),
        ],
        [
            ("qasper_abstractive_qa", "bleu,extract-json"),
            ("evidence_inference", "bleu"),
        ],
        [
            ("qasper_abstractive_qa", "rouge,extract-json", "rougeL"),
            ("qasper_abstractive_qa", "rougeL"),
        ],
        [
            ("qasper_abstractive_qa", "f1_token,extract-json"),
            ("qasper_abstractive_qa", "f1_answer"),
        ],
        [
            ("qasper_abstractive_qa", "f1_evidence_exact,extract-json", "f1"),
            ("qasper_abstractive_qa", "f1_evidence"),
        ],
        [("scierc_re", "bleu,extract-json"), ("scierc_re", "bleu")],
        [("scierc_re", "rouge,extract-json", "rougeL"), ("scierc_re", "rougeL")],
        [
            ("scierc_re", "f1_relation_exact,extract-json", "f1"),
            ("scierc_re", "f1_rel"),
        ],
        [("scifact_entailment", "bleu,extract-json"), ("evidence_inference", "bleu")],
        [
            ("scifact_entailment", "rouge,extract-json", "rougeL"),
            ("scifact_entailment", "rougeL"),
        ],
        [
            ("scifact_entailment", "label_accuracy,extract-json"),
            ("scifact_entailment", "acc_label"),
        ],
        [
            ("scifact_entailment", "f1_evidence_exact,extract-json", "f1"),
            ("scifact_entailment", "f1_evidence"),
        ],
        [
            ("mup_single_document_summarization", "bleu,none"),
            ("mup_single_document_summarization", "bleu"),
        ],
        [
            ("mup_single_document_summarization", "rouge,none", "rougeL"),
            ("mup_single_document_summarization", "rougeL"),
        ],
        [("bioasq_list_qa", "bleu,extract-json"), ("evidence_inference", "bleu")],
        [
            ("bioasq_list_qa", "rouge,extract-json", "rougeL"),
            ("bioasq_list_qa", "rougeL"),
        ],
        [
            ("bioasq_list_qa", "f1_list_exact,extract-json", "f1"),
            ("bioasq_list_qa", "f1_answer"),
        ],
    ]

    metrics_dir = paths.EVAL_DIR / "results/metrics"

    res = []

    for fname in metrics_dir.iterdir():
        # Skip `_old` directory with results we don't need.
        if fname.is_dir():
            continue

        model = fname.stem

        results = json.load(open(fname))
        for key_tup, (task_name, metric_name) in lookup:
            the_val = get_nested_value(results, key_tup)
            to_append = {
                "model": model,
                "task": task_name,
                "metric": metric_name,
                "value": the_val,
            }
            res.append(to_append)

    res = pd.DataFrame(res)
    res["task"] = [
        task if task == "evidence_inference" else task.split("_")[0]
        for task in res["task"]
    ]

    df = res.pivot_table(index=["task", "metric"], columns="model", values="value")
    by_metric = df.groupby("metric").mean()

    overall = by_metric.apply(["mean", "median"], axis=0)

    # Metrics.
    df_to_write = df.reset_index()
    by_metric_to_write = by_metric.reset_index()
    by_metric_to_write.insert(0, "task", "overall")

    overall_to_write = overall.reset_index().rename(columns={"index": "metric"})
    overall_to_write.insert(0, "task", "overall")

    to_write = pd.concat(
        [df_to_write, by_metric_to_write, overall_to_write], axis=0
    ).set_index(["task", "metric"])
    to_write.columns.name = None
    to_write = put_baselines_last(to_write.T)

    to_write.to_csv(
        "results/tables/metrics_science_all.tsv", sep="\t", float_format="%0.2f"
    )

    # Subset of task / metric combos that we know are reliable.
    reliable_task_metrics = [
        ("bioasq", "f1_answer"),
        ("biored", "f1_ner"),
        ("evidence_inference", "bleu"),
        ("multicite", "f1_label"),
        ("mup", "rougeL"),
        ("qasper", "f1_answer"),
        ("scierc", "bleu"),
        ("scifact", "acc_label"),
        ("scifact", "f1_evidence"),
    ]

    good_subset = df.loc[reliable_task_metrics]
    good_summary = good_subset.apply(["mean", "median"], axis=0)
    good_to_write = good_subset.reset_index()
    summary_to_write = good_summary.reset_index().rename(columns={"index": "metric"})
    summary_to_write.insert(0, "task", "science")
    to_write = pd.concat([good_to_write, summary_to_write], axis=0).set_index(
        ["task", "metric"]
    )
    to_write.columns.name = None
    to_write = put_baselines_last(to_write.T)

    to_write.to_csv(
        "results/tables/metrics_science_reliable.tsv", sep="\t", float_format="%0.2f"
    )

    return to_write


########################################


# Open instruct.


def mean(xs):
    return sum(xs) / len(xs)


def json_mean(result_dir, fname):
    d = json.load(open(result_dir / fname))
    return mean(d.values())


def json_field(result_dir, fname, field):
    d = json.load(open(result_dir / fname))
    return d[field]


def process_mmlu(result_dir):
    all_scores = []
    for subdomain in result_dir.iterdir():
        df = pd.read_csv(subdomain)
        all_scores.append(df["correct"].mean())

    return mean(all_scores)


def process_tydiqa(result_dir):
    d = json.load(open(result_dir / "metrics.json"))
    scores = [entry["f1"] for entry in d.values()]
    return mean(scores)


def process_alpaca(result_dir):
    expected_file = Path(result_dir / "metrics.json")
    if expected_file.exists():
        d = json.load(open(result_dir / "metrics.json"))
        win_rate = d["win_rate"]["model-greedy-long"] / 100
        length = d["avg_length"]["model-greedy-long"]
        return win_rate, length
    else:
        return -1


def process_one_model(model_dir):
    agg_lookup = {
        "bbh_cot": partial(json_mean, fname="metrics.json"),
        "bbh_direct": partial(json_mean, fname="metrics.json"),
        "codex_eval_temp_0.1": partial(
            json_field, fname="metrics.json", field="pass@1"
        ),
        "codex_eval_temp_0.8": partial(
            json_field, fname="metrics.json", field="pass@10"
        ),
        "gsm_cot": partial(json_field, fname="metrics.json", field="exact_match"),
        "gsm_direct": partial(json_field, fname="metrics.json", field="exact_match"),
        "mmlu_0shot": partial(json_field, fname="metrics.json", field="average_acc"),
        "mmlu_5shot": partial(json_field, fname="metrics.json", field="average_acc"),
        "toxigen": partial(json_mean, fname="metrics.json"),
        "tydiqa_goldp_1shot": process_tydiqa,
        "tydiqa_no_context_1shot": process_tydiqa,
        "alpaca_eval": process_alpaca,
    }

    res = {"name": model_dir.name}
    for subdir in model_dir.iterdir():
        agg_fn = agg_lookup[subdir.name]
        if subdir.name == "alpaca_eval":
            res["alpaca_win_rate"], res["alpaca_length"] = agg_fn(subdir)
        else:
            res[subdir.name] = agg_fn(subdir)

    return res


def get_open_instruct_results():
    metrics_dir = paths.EVAL_DIR / "results/metrics_open_instruct"
    res = []

    for model_dir in metrics_dir.iterdir():
        metrics = process_one_model(model_dir)
        res.append(metrics)

    res = pd.DataFrame(res).set_index("name").T

    # Use the metrics reported in the tulu 2 paper.
    metrics_to_keep = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "tydiqa_goldp_1shot",
        "codex_eval_temp_0.8",
        "toxigen",
        "alpaca_win_rate",
        "alpaca_length"
    ]
    metric_names = [
        "exact_match",
        "exact_match",
        "exact_match",
        "f1",
        "pass@10",
        "overall",
        "win_rate",
        "length"
    ]

    metrics_reliable = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "codex_eval_temp_0.8",
        "alpaca_win_rate",
    ]

    res = res.loc[metrics_to_keep].reset_index()
    res = res.rename(columns={"index": "task"})
    res.insert(1, "metric", metric_names)
    res.columns.name = None
    res = res.set_index(["task", "metric"])

    summary = (
        res.loc[metrics_reliable]
        .apply(["mean", "median"], axis=0)
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    summary.insert(0, "task", "open_instruct")
    summary = summary.set_index(["task", "metric"])
    to_write = pd.concat([res, summary], axis=0)
    to_write = to_write.T

    to_write.to_csv(
        paths.EVAL_DIR / "results/tables/metrics_open_instruct.tsv",
        sep="\t",
        float_format="%0.2f",
    )

    return to_write


########################################

# Putting it all together


def main():
    results_science = get_science_results()
    results_open_instruct = get_open_instruct_results()

    results = results_science.merge(
        results_open_instruct, left_index=True, right_index=True, how="left"
    )
    results = put_baselines_last(results)
    results.to_csv(
        paths.EVAL_DIR / "results/tables/metrics.tsv", sep="\t", float_format="%0.2f"
    )


if __name__ == "__main__":
    main()
