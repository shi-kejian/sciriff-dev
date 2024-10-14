"""
Simple convenience script to pretty-print generated instances.

Might make this fancier at some point.
"""

import argparse
from lib import util
import json
from random import shuffle


def preview(
    instance_file, num_instances=1, format_json=False, template_ids=None, random=False
):
    "Preview the first `num_instances` instances of a task."
    data = util.load_jsonl(instance_file)
    if random:
        shuffle(data)
    for i, instance in enumerate(data):
        if i == num_instances:
            break
        if template_ids is not None and instance["_template_id"] not in template_ids:
            # Skip if not doing this template.
            continue
        print("INPUT\n")
        print(instance["input"])
        print("\n" + "-" * 20 + "\n")
        print("OUTPUT\n")
        output = instance["output"]
        if format_json:
            output = json.dumps(json.loads(output), indent=2)

        print(output)
        print("\n" + "-" * 80 + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Preview generated instances.")
    parser.add_argument("--instance_file", type=str, help="The file to preview")
    parser.add_argument(
        "--n_instances", type=int, default=1, help="Number of instances to show."
    )
    parser.add_argument(
        "--templates",
        type=util.comma_separated_list_arg,
        default="0",
        help="A comma-separated list of template ID's to preview. By default, do the first.",
    )
    parser.add_argument(
        "--format_json",
        action="store_true",
        help="If given, pretty-print JSON-formatted output.",
    )
    parser.add_argument(
        "--random", action="store_true", help="If given, choose random instances."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    preview(
        args.instance_file,
        args.n_instances,
        args.format_json,
        args.templates,
        args.random,
    )
