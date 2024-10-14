"""
Input conversion util codes for open-instruct. Adapted from https://github.com/allenai/open-instruct.
"""

import random

# Input encoding template.
# For v1 we use a subset of the templates from open-instruct, for Science-adapt instances.
encoding_templates_wo_input = [
    ("{instruction}\n", "{output}", 0.3), # Upweight this template
    ("{instruction}\n", "{output}", 0.1),
    ("{instruction}", "\n{output}", 0.1),
    ("{instruction} Output:", "{output}", 0.05), # To complement. Science-adapt instances don't have demarcation text "Output:"
    ("{instruction}\nResponse:", "{output}", 0.05),
    ("{instruction}\n\nAnswer:", "{output}", 0.05),
    ("Task: {instruction}\n\n", "{output}", 0.05),
    ("Instruction: {instruction}\n", "{output}", 0.05),
    ("Instruction: {instruction}\nOutput:", "{output}", 0.05),
    ("You need to complete the following task:\n\n{instruction}\n\n", "{output}", 0.03),
    ("Can you help with this?\n\n{instruction}\n", "{output}", 0.02),
    ("Plase answer the following request: {instruction}\nAnswer:", "{output}", 0.02),
    ("Tell me how would you respond to the following request.\n{instruction}\n", "{output}", 0.03),
    ("Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:", "{output}", 0.1), # Alpaca template
]


def encode_instruction_example(instruction, input, output, random_template=True, eos_token=None):
    if random_template:
            # randomly choose a template without additional input field
            prompt_template, completion_template, _ = random.choices(
                encoding_templates_wo_input, weights=[w for _, _, w in encoding_templates_wo_input]
            )[0]
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output.strip())
    else:
        if input is not None and input.strip() != "":
            prompt = instruction.strip() + "\n\n" + input.strip() + "\n\n"
            completion = output.strip()
        else:
            prompt = instruction.strip() + "\n\n"
            completion = output.strip()

    data = {
        "prompt": prompt,
        "completion": completion + eos_token if eos_token else completion,
    }
    return data


def encode_few_shot_example(instruction, examplars, input, output, eos_token=None):
    prompt = instruction.strip() + "\n\n"
    for examplar in examplars:
        prompt += "Input:\n" + examplar["input"].strip() + "\n"
        prompt += "Output:\n" + examplar["output"].strip() + "\n\n"

    prompt += "Input:\n" + input.strip() + "\n"
    prompt += "Output:\n"

    data = {
        "prompt": prompt,
        "completion": output.strip() + eos_token if eos_token else output.strip(),
    }
    return data