"""
Some quick spot-checks to make sure the mixtures look reasonable.

Usage: pytest test_mixtures.py
"""

import create_mixture
import pandas as pd
from collections import Counter

from lib import util, paths

data_dir = paths.project_root / "data/training_mixtures"


def get_science_counts(ids):
    science_counts = Counter()
    for id in ids:
        if "science" in id:
            dataset = id.split(".")[1]
            science_counts[dataset] += 1

    return pd.Series(science_counts)


def check_tulu(ids, settings):
    "Check that the amount of tulu data looks right."
    is_science = pd.DataFrame(["science" in id for id in ids])
    if settings["tulu_data"] == "none":
        # No tulu data.
        assert is_science.all().item()
    elif settings["tulu_data"] == "match":
        # Same amount of tulu as science.
        assert is_science.sum().item() == (~is_science).sum().item()
    elif settings["tulu_data"] == "all":
        # Should be at least 100k non-science insts.
        assert (~is_science).sum().item() > 100000
    else:
        raise Exception("Unexpected tulu data setting.")


def check_science(ids, science_counts, settings):
    "Make sure the number of science instances isn't more than expected."
    if settings["science_insts"] == 0:
        # Should be no science instances.
        assert sum(["science" in id for id in ids]) == 0
    else:
        # Shouldn't have more instructions per task than the max.
        assert science_counts.max() <= settings["science_insts"]


def check_eval(science_counts, settings):
    "Make sure we don't have eval tasks when we shouldn't and vice versa."
    if settings["science_insts"] == 0:
        # If no science, don't need to check.
        return

    eval_tasks = create_mixture.get_eval_tasks()
    shared = set(eval_tasks) & set(science_counts.keys())
    if settings["use_eval"] == "yes":
        # All eval tasks should be there.
        assert len(shared) == len(eval_tasks)
    elif settings["use_eval"] == "no":
        # No eval tasks should be there.
        assert len(shared) == 0
    else:
        raise Exception("Unexpected value for `use_eval`.")


def check_format(data):
    "Make sure all the science instances are formatted correctly."
    for inst in data:
        if "science" not in inst["id"]:
            continue

        messages = inst["messages"]
        assert len(messages) == 2
        expected_keys = set(["role", "content"])
        assert set(messages[0].keys()) == expected_keys
        assert set(messages[1].keys()) == expected_keys
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


def test_mixtures():
    # Test to check all created mixtures.
    for subdir in data_dir.iterdir():
        for train_file in subdir.iterdir():
            splt = train_file.stem.split("_")
            settings = {
                "tulu_data": splt[1],
                "science_insts": int(splt[3]),
                "use_eval": splt[5],
            }
            data = util.load_jsonl(train_file)
            ids = [entry["id"] for entry in data]
            science_counts = get_science_counts(ids)
            check_tulu(ids, settings)
            check_science(ids, science_counts, settings)
            check_eval(science_counts, settings)
            check_format(data)
