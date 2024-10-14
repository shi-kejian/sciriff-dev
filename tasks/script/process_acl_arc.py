import json
import random
from pathlib import Path

# import pandas as pd


# Iterate over each dataset entry and generate prompt templates
def conver_template(dataset, file_name, split, quota_to_template):
    prompt_templates = {}

    for data_entry_id in range(len(dataset)):
        text = dataset[data_entry_id]["extended_context"]
        cited_paper_title = dataset[data_entry_id]["cited_paper_title"]
        cited_author_ids = "".join(dataset[data_entry_id]["cited_author_ids"])

        false_id = 0
        while false_id == data_entry_id:
            false_id = random.randint(0, len(dataset))
        false_cited_paper_title = "".join(dataset[false_id]["cited_paper_title"])

        if data_entry_id < quota_to_template:
            template_name = "_ir_mixed"
            _ir_mixed = {
                "instruction": "",
                "prompt_template": f"Based on the given text:{text}, please provide me the cited authors.",
                "output": "The cited authors are " + cited_author_ids,
            }
            if template_name not in prompt_templates.keys():
                prompt_templates[template_name] = []
            prompt_templates[template_name].append(_ir_mixed)

        if data_entry_id >= quota_to_template and data_entry_id < quota_to_template * 2:
            template_name = "_ir_instruction"
            _ir_instruction = {
                "instruction": "You are a scientist who is interested in the following paper. Please complete the following tasts.",
                "prompt_template": f"According to the following text'{text}', which author(s) is/are cited in the paper?",
                "output": "The cited authors are " + cited_author_ids,
            }
            if template_name not in prompt_templates.keys():
                prompt_templates[template_name] = []
            prompt_templates[template_name].append(_ir_instruction)

        if (
            data_entry_id >= quota_to_template * 2
            and data_entry_id < quota_to_template * 3
        ):
            template_name = "_nli_true_answers"
            _nli_true_answers = {
                "instruction": "You are a scientist who is interested in the following paper. Please complete the following tasts.",
                "prompt_template": f"Determine whether the following statement is true or not.\nAccording to the following text '{text}', the content '{cited_paper_title}' is related to the text.",
                "output": "The statement is true",
            }
            if template_name not in prompt_templates.keys():
                prompt_templates[template_name] = []
            prompt_templates[template_name].append(_nli_true_answers)

        if data_entry_id >= quota_to_template * 3:
            template_name = "_nli_false_answers"
            _nli_false_answers = {
                "instruction": "You are a scientist who is interested in the following paper. Please complete the following tasts.",
                "prompt_template": f"Determine whether the following statement is true or not.\nAccording to the following text '{text}', the content '{false_cited_paper_title}' is related to the text.",
                "output": "The statement is false",
            }
            if template_name not in prompt_templates.keys():
                prompt_templates[template_name] = []
            prompt_templates[template_name].append(_nli_false_answers)

    for key in prompt_templates.keys():
        print(
            f"Number of prompt templates for {file_name}{key}: {len(prompt_templates[key])}"
        )
        output_name = output_path / (file_name + split + key + ".jsonl")

        # Save the prompt templates as a .jsonl file
        with open(output_name, "w") as jsonl_file:
            for prompt_template in prompt_templates[key]:
                json.dump(prompt_template, jsonl_file)
                jsonl_file.write("\n")
        # Save the prompt templates as a .csv file
        # df = pd.DataFrame(prompt_templates[key])
        # df.to_csv(output_name + ".csv", index=False)


if __name__ == "__main__":
    file_name = "ACL-ARC"
    file_dict = {
        "_train": Path("data/downloads/acl-arc/train.jsonl"),
        "_dev": Path("data/downloads/acl-arc/dev.jsonl"),
        "_test": Path("data/downloads/acl-arc/test.jsonl"),
    }
    output_path = Path("data/processed/acl_arc/")
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = []
    prompt_templates = [
        "_ir_mixed",
        "_ir_instruction",
        "_nli_true_answers",
        "_nli_false_answers",
    ]

    split_dict = {value: key for key, value in file_dict.items()}
    for file_path in file_dict.values():
        # print("Preprocessing file: ", file_path)
        dataset = []
        for line in open(file_path, "r", encoding="utf-8"):
            dataset.append(json.loads(line))
        prompt_templates_len = len(prompt_templates)
        quota_to_template = len(dataset) // prompt_templates_len
        split = split_dict[file_path]
        conver_template(dataset, file_name, split, quota_to_template)
