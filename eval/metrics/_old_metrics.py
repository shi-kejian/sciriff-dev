import json
import re
import string
from collections import Counter
import numpy as np

from lib import util


def dict_to_tuples(ent_dict):
    "Convert from a list of dicts to a list of (entity, type) tuples."
    res_typed = set()
    res_untyped = set()

    for ent_type, mentions in ent_dict.items():
        # Sometimes the model outputs a nested list of mentions by mistake; just flatten
        # these.
        mentions = util.flatten(mentions)
        for mention in mentions:
            try:
                # Try to normalize the mention and add it to predictions.
                mention_normalized = mention.lower().strip()
            except AttributeError:
                # If the model output an unexpected type (e.g. None or a dict), then
                # just skip it.
                continue
            else:
                # If it worked, add the mention to the predictions.
                res_typed.add((mention_normalized, ent_type))
                res_untyped.add(mention_normalized)

    return res_typed, res_untyped


def unpack(predictions, references):
    if len(predictions) != 1 or len(references) != 1:
        raise ValueError("Unexpected number of predictions/references.")

    return json.loads(predictions[0]), json.loads(references[0])


def agg_f1_loop(items, name):
    """
    Pull together counts from all instances and compute F1.
    """
    preds = sum(item[f"preds_{name}"] for item in items)
    refs = sum(item[f"refs_{name}"] for item in items)
    correct = sum(item[f"correct_{name}"] for item in items)

    precision = correct / preds if preds > 0 else 0
    recall = correct / refs
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    res = {f"p_{name}": precision, f"r_{name}": recall, f"f1_{name}": f1}
    return res


def agg_f1_ner_exact(items):
    """
    Compute typed and untyped F1.
    """
    res = {}

    for name in ["typed", "untyped"]:
        res_loop = agg_f1_loop(items, name)
        res.update(res_loop)

    return res


def f1_ner_exact(predictions, references):
    """
    Counts the number of matches for each instance; final metric computation is done by
    the aggregation function.
    """
    counts_typed = {"correct": 0, "refs": 0, "preds": 0}
    counts_untyped = {"correct": 0, "refs": 0, "preds": 0}

    for pred, ref in zip(predictions, references):
        pred_typed, pred_untyped = dict_to_tuples(pred)
        ref_typed, ref_untyped = dict_to_tuples(ref)

        counts_typed["correct"] += len(pred_typed & ref_typed)
        counts_typed["refs"] += len(ref_typed)
        counts_typed["preds"] += len(pred_typed)

        counts_untyped["correct"] += len(pred_untyped & ref_untyped)
        counts_untyped["refs"] += len(ref_untyped)
        counts_untyped["preds"] += len(pred_untyped)

    f1_typed = compute_f1(
        counts_typed["correct"], counts_typed["preds"], counts_typed["refs"]
    )
    f1_untyped = compute_f1(
        counts_untyped["correct"], counts_untyped["preds"], counts_untyped["refs"]
    )

    return {"f1_typed": f1_typed, "f1_untyped": f1_untyped}


def label_accuracy(predictions, references, field):
    """
    F1 score on label and evidence for SciFact.
    """
    preds_dict, refs_dict = unpack(predictions, references)

    return preds_dict[field].lower().strip() == refs_dict[field].lower().strip()


def agg_f1(items):
    """
    Pull together counts from all instances and compute F1.
    """
    # TODO(wadden) This is really similar to `agg_f1_loop`; refactor.
    preds = sum(item["preds"] for item in items)
    refs = sum(item["refs"] for item in items)
    correct = sum(item["correct"] for item in items)

    precision = correct / preds if preds > 0 else 0
    recall = correct / refs
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    res = {"p": precision, "r": recall, "f1": f1}
    return res


def f1_evidence_exact(predictions, references, field):
    """
    F1 score on evidence for SciFact, Qasper, etc.
    """

    def normalize(xs):
        # Each entry in the evidence should be a string; if not, just skip it.
        return [x.strip().lower() for x in xs if isinstance(x, str)]

    preds_dict, refs_dict = unpack(predictions, references)

    evs_pred = set(normalize(preds_dict[field]))
    evs_ref = set(normalize(refs_dict[field]))

    return {
        "correct": len(evs_pred & evs_ref),
        "preds": len(evs_pred),
        "refs": len(evs_ref),
    }


def normalize_list_entry(entry):
    "Convert to lowercase and convert numbers to strings."
    if entry is None:
        return ""
    elif isinstance(entry, str):
        return entry.strip().lower()
    elif isinstance(entry, int):
        return str(entry)
    elif isinstance(entry, float):
        return str(round(entry, 2))
    else:
        raise TypeError("Unexpected type for list entry.")


def f1_relation_exact(predictions, references):
    """
    Exact F1 score on relations tuples. Requires all spans and relation to match
    exactly. This is extremely strict; will need to find a way to do a softer match.
    """

    def normalize(tups):
        # Convert each relation tuple into a tuple of normalized strings.
        # Throw out any cases where the entries aren't lists

        # TODO(davidw) We need to have a count for how often this doesn't work.
        good_tups = [tup for tup in tups if isinstance(tup, list)]

        res = set()
        for tup in good_tups:
            # TODO(wadden) Check if there are any None entries in gold.
            try:
                to_add = tuple([normalize_list_entry(entry) for entry in tup])
                res.add(to_add)
            except TypeError:
                # This will happen if a model gets the nesting structure wrong and
                # returns a nested list; if this happens, just skip.
                continue

        return res

    counts_typed = {"correct": 0, "refs": 0, "preds": 0}
    counts_untyped = {"correct": 0, "refs": 0, "preds": 0}

    for pred, ref in zip(predictions, references):
        pred_set = normalize(pred)
        ref_set = normalize(ref)

        import ipdb; ipdb.set_trace()



def compute_f1(correct, preds, refs):
    precision = correct / preds if preds > 0 else 0
    recall = correct / refs if refs > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return {"p": precision, "r": recall, "f1": f1}


def f1_list_exact(preds, refs):
    """
    Exact F1 score on list-valued predictions.
    """

    def normalize(xs):
        try:
            return [normalize_list_entry(x) for x in xs]
        except TypeError:
            return [""]

    counts = {"correct": 0, "preds": 0, "refs": 0}

    for pred, ref in zip(preds, refs):
        pred_set = set(normalize(pred))
        ref_set = set(normalize(ref))
        counts["correct"] += len(pred_set & ref_set)
        counts["preds"] += len(pred_set)
        counts["refs"] += len(ref_set)

    f1 = compute_f1(counts["correct"], counts["preds"], counts["refs"])
    return f1


####################

# Qasper token F1.


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # There are some cases where the answer might be a number.
        if text is None:
            return ""
        elif isinstance(text, str):
            return text.lower()
        elif isinstance(text, int):
            return str(text)
        elif isinstance(text, float):
            return str(round(text, 2))
        elif isinstance(text, list):
            # If it's a list, flatten and then join the elements into a string.
            return " ".join(util.flatten(text)).lower()
        elif isinstance(text, dict):
            # If it's a dict, just convert to a string.
            return str(text)
        else:
            raise TypeError("Unexpected type for answer.")

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_token(predictions, references, field):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    preds, refs = unpack(predictions, references)
    prediction_tokens = normalize_answer(preds[field]).split()
    references_tokens = normalize_answer(refs[field]).split()
    common = Counter(prediction_tokens) & Counter(references_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(references_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


####################


def agg_dict(items):
    """
    Keywise aggregation of dicts; useful for metrics like ROUGE that return a dict.
    """
    the_keys = items[0].keys()

    all_scores = {k: [] for k in the_keys}
    for item in items:
        for k, v in item.items():
            all_scores[k].append(v)

    res = {k: np.mean(v) for k, v in all_scores.items()}

    return res
