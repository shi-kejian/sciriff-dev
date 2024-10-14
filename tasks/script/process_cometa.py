import json
from pathlib import Path
import pandas as pd


# Iterate over each dataset entry and generate prompt templates
def conver_template(dataset, file_name, split, quota_to_template):
    prompt_templates = {}
    for data_entry_id in range(len(dataset)):
        text = dataset[data_entry_id]["Example"]
        entity = dataset[data_entry_id]["General SNOMED Label"]

        if data_entry_id < quota_to_template:
            template_name = "_ner_mixed"
            _ner_mixed = {
                "instruction": "",
                "prompt_template": f"Based on the given text:'{text}', please provide me the most appropriate health terms related to the text.",
                "output": "The health terms is/are " + str(entity),
            }
            if template_name not in prompt_templates.keys():
                prompt_templates[template_name] = []
            prompt_templates[template_name].append(_ner_mixed)

        if data_entry_id >= quota_to_template and data_entry_id < quota_to_template * 2:
            template_name = "_ner_instruction"
            _ner_instruction = {
                "instruction": "You are an expert in medical and health science. Please complete the following tasks.",
                "prompt_template": f"Text:'{text}'\nPlease provide me the most appropriate health terms related to the text.",
                "output": "The health terms is/are " + str(entity),
            }
            if template_name not in prompt_templates.keys():
                prompt_templates[template_name] = []
            prompt_templates[template_name].append(_ner_instruction)

    for key in prompt_templates.keys():
        print(
            f"Number of prompt templates for {file_name}{key}: {len(prompt_templates[key])}"
        )
        output_name = output_path / f"{file_name}{split}{key}.jsonl"

        # Save the prompt templates as a .jsonl file
        with open(output_name, "w") as jsonl_file:
            for prompt_template in prompt_templates[key]:
                json.dump(prompt_template, jsonl_file)
                jsonl_file.write("\n")
        # Save as csv file
        # df = pd.DataFrame(prompt_templates[key])
        # df.to_csv(output_name + ".csv", index=False)


if __name__ == "__main__":
    file_name = "cometa"
    file_dict = {
        "_train": Path("data/downloads/cometa/splits/random/train.csv"),
        "_dev": Path("data/downloads/cometa/splits/random/dev.csv"),
        "_test": Path("data/downloads/cometa/splits/random/test.csv"),
    }
    output_path = Path("data/processed/cometa")
    dataset = []
    prompt_templates = ["_ner_mixed", "_ner_instruction"]

    split_dict = {value: key for key, value in file_dict.items()}
    for file_path in file_dict.values():
        # print("Preprocessing file: ", file_path)
        dataset = pd.read_csv(file_path, sep="\t")
        dataset = dataset.to_dict("records")
        prompt_templates_len = len(prompt_templates)
        quota_to_template = len(dataset) // prompt_templates_len
        split = split_dict[file_path]
        conver_template(dataset, file_name, split, quota_to_template)
