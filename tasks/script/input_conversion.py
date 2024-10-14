import json
import random
import argparse
from pathlib import Path
from instruction_encode_templates import encode_instruction_example, encode_few_shot_example

def convert_super_ni_data(data_dir, output_dir, zero_shot_examples_per_task=60, few_shot_examples_per_task=20, n_few_shot=2):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tasks = []
    with open(data_dir / "splits/xlingual/train_tasks.txt", "r") as fin:
        for line in fin:
            if not "_mmmlu_" in line:   # skip mmlu to avoid test leakage
                train_tasks.append(line.strip())

    with open(output_dir / "super_ni_data.jsonl", "w") as fout:
        for task in train_tasks:
            task_file = data_dir / f"tasks/{task}.json"
            with open(task_file, "r") as fin:
                task_data = json.load(fin)
            instruction = task_data["Definition"][0]
            if zero_shot_examples_per_task + few_shot_examples_per_task < len(task_data["Instances"]):
                instances = random.sample(task_data["Instances"], k=zero_shot_examples_per_task+few_shot_examples_per_task)
            else:
                instances = task_data["Instances"]
            for instance in instances[:zero_shot_examples_per_task]:
                encoded_example = encode_instruction_example(
                    instruction=instruction, 
                    input=instance["input"], 
                    output=instance["output"][0],
                    random_template=True,
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "super_ni",
                    "id": f"super_ni_{instance['id']}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
            for instance in instances[zero_shot_examples_per_task:]:
                if n_few_shot < len(task_data["Positive Examples"]):
                    examplars = random.sample(task_data["Positive Examples"], k=n_few_shot)
                else:
                    examplars = task_data["Positive Examples"]
                encoded_example = encode_few_shot_example(
                    instruction=instruction,
                    examplars=examplars,
                    input=instance["input"],
                    output=instance["output"][0],
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "super_ni",
                    "id": f"super_ni_{instance['id']}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")

def convert_flan_v2_data(data_dir, output_dir, data_file="tulu_v1_resampled_flan_100k.jsonl"):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(data_dir / data_file, "r") as fin:
        for line in fin:
            examples.append(json.loads(line))

    with open(output_dir / "flan_v2_data.jsonl", "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            fout.write(json.dumps({
                "dataset": "flan_v2",
                "id": f"flan_v2_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }) + "\n")


def convert_science_data(data_dir, output_dir, num_examples=None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(data_dir / "science_train.jsonl", "r") as fin:
        for line in fin:
            examples.append(json.loads(line))

    if num_examples:
        examples = random.sample(examples, k=num_examples)

    with open(output_dir / "science_data.jsonl", "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": f"science.{example['dataset']}",
                "id": f"science_{idx}",
                "messages": [
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]}
                ],
            }) + "\n")


def convert_science_adapt_data(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents_exist_ok=True)

    examples = []
    with open(data_dir / "science_adapt_train.jsonl", "r") as fin:
        for line in fin:
            examples.append(json.loads(line))

    with open(output_dir / "science_adapt_data.jsonl", "w") as fout:
        for idx, example in enumerate(examples):
            instruction = example["input"]
            output = example["output"]

            encoded_example = encode_instruction_example(
                instruction=instruction,
                input=None,  # No separate input in our data per open-instruct definition
                output=output,
                random_template=True,
                eos_token=None
            )

            fout.write(json.dumps({
                "dataset": "science_adapt",
                "id": f"science_adapt_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")

if __name__ == "__main__":
    # Now we don't care about other datasets. Just convert science_adapt.
    supported_datasets = ["science_adapt", "super_ni", "flan_v2", "science"]

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--raw_data_dir", type=str, default="./tasks/mixtures/")
    arg_parser.add_argument("--output_dir", type=str, default="./tasks/mixtures_processed/")
    arg_parser.add_argument("--combine_datasets", action="store_true")
    arg_parser.add_argument("--datasets", nargs='*', default=["science_adapt"], help=f"List of datasets to process. Supported: {supported_datasets}")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()
    random.seed(args.seed)

    for dataset in args.datasets:
        if dataset in supported_datasets:
            globals()[f"convert_{dataset}_data"](Path(args.raw_data_dir) / dataset, Path(args.output_dir) / dataset)


    if args.combine_datasets and len(args.datasets) > 1:
        combined_output_path = Path(args.output_dir) / "combined_data.jsonl"
        with open(combined_output_path, "w") as fout:
            for dataset in args.datasets:
                dataset_output_file = Path(args.output_dir) / dataset / f"{dataset}_data.jsonl"
                with open(dataset_output_file, "r") as fin:
                    for line in fin:
                        fout.write(line)

        print(f"Combined mixture created, to {combined_output_path}")

