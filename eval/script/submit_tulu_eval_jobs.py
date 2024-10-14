# NOTE: Copied from `open-instruct/scripts/submit_eval_jobs.py`.

import copy
import subprocess
import yaml
import random
import re
import itertools
from datetime import date
from pathlib import Path

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_eval.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

cluster = "ai2/s2-cirrascale-l40"
num_gpus = 1
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "high"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

# modify here for different set of experiments
experiment_groups = [
    "mmlu_0shot",
    "mmlu_5shot",
    "gsm_direct",
    "gsm_cot",
    "bbh_direct",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "tydiqa_no_context_1shot",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    # "truthfulqa",
    "toxigen",
    "alpaca_eval",
]

# model to evaluate, each in the followng format: model name, their beaker id, checkpoint subfolder
models = [
    # Single-stage finetuning.
    ("science_adapt_1-stage_tuluv2_no_science", "01HKG46RNVAP3NSHNDH019R5KB", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_balance_task_80k", "01HKEN0TEZ6CWS3PZMAFNVQMM7", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_balance_task_80k_with_eval", "01HKKTE7J19AZW5VTQN3ACPMY4", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_balance_task_40k", "01HKN3141TNYP5DABQ2D5D4MHC", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_balance_task_5k", "01HN9TKDPE3SBQCW6GRVS0BY72", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_per_task_200", "01HN9TKDW2CP5BK91YHMQS9145", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_balance_task_10k", "01HNF4VNF8C7159M64B10E99FH", None, "tuned_lm"),
    # ("science_adapt_continued_4k_per_task_200", "01HN9PPJGZR13M0PY7ZEQRX26D", None, "tuned_lm"),
    # ("science_adapt_continued_4k_per_task_200_tulu_ratio_1", "01HNK9MEQJD9V00NMWW2XM2RFV", None, "tuned_lm"),
    # ("merge-science_adapt_continued_4k_per_task_200-tuluv2_no_science", "01HNM72NEM9YGT3A8V1E40JQ42", None, "tuned_lm"),
    # ("merge-science_adapt_continued_4k_per_task_200_tulu_ratio_1-tuluv2_no_science", "01HNM75ZG00WV22RD88JXEVF47", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_per_task_500", "01HNPW2TP7PJGRJ6HK107XFXVN", None, "tuned_lm"),
    # ("science_adapt_1-stage_4k_per_task_1000", "01HNPW2TTRXC50XY017J3Q48R7", None, "tuned_lm"),

    # Tulu model from paper
    ("tulu_v2_7B_jax", "01HBXTF305QARZ7P4T6ASXXVAM", None, "tuned_lm"),

    # our ablation models
    # ("finetuned_llama1_7B_dolly", "01GZVKGQZAMQMVG9307KWS4GMN", None, "tuned_lm"),
    # ("finetuned_llama1_7B_flan_v2", "01GZVKGR5DW1SXXWSMWE2QYWYR", None, "tuned_lm"),
    # ("finetuned_llama1_7B_cot", "01GZVKGRA3X4SYQF1PZ29DSZFE", None, "tuned_lm"),
    # ("finetuned_llama1_7B_code_alpaca", "01GZVKGREPDJ6FZM3S4B0J8VB9", None, "tuned_lm"),
    # ("finetuned_llama1_7B_baize", "01GZVKGRKAHJW2AK3ZF88G13HA", None, "tuned_lm"),
    # ("finetuned_llama1_7B_oasst1", "01GZVKGRQZ4359W31CAEHWFVSB", None, "tuned_lm"),
    # ("finetuned_llama1_7B_gpt4_alpaca", "01GZVKGRWJ2VVCXY5KP46814JP", None, "tuned_lm"),
    # ("finetuned_llama1_7B_super_ni", "01GZVKGS1S527GYKRA4Y26ZP5S", None, "tuned_lm"),
    # ("finetuned_llama1_7B_self_instruct", "01GZVKGS7JTYK0M35AFXHY0CD0", None, "tuned_lm"),
    # ("finetuned_llama1_7B_stanford_alpaca", "01GZVKGSHNPRFSJBS4K74FTRDC", None, "tuned_lm"),
    # ("finetuned_llama1_7B_unnatural_instructions", "01GZVKGSP9BAW8XTWB9509SPDB", None, "tuned_lm"),
    # ("finetuned_llama1_7B_sharegpt", "01GZWDNED8KP28SAR1159WZ366", None, "tuned_lm"),
]

#--------------- experiments about number of supervision tasks -------------------------

# for experiment_group, model_info in itertools.product(experiment_groups, models):
for model_info, experiment_group in itertools.product(models, experiment_groups):
    d = copy.deepcopy(d1)

    model_name = model_info[0] + f"_{model_info[2]}" if model_info[2] is not None else model_info[0]
    name = f"open_instruct_eval_{experiment_group}_{model_name}_{today}"
    d['description'] = f"Open instruct evaluation for model {model_name}, task {experiment_group}"
    d['tasks'][0]['name'] = name

    out_dir = Path("/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/eval/results/metrics_open_instruct") / model_name / experiment_group
    if out_dir.exists():
        continue

    print(f"Submitting {experiment_group} for model: {model_info[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if experiment_group == "mmlu_0shot":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.mmlu.run_eval \
            --ntrain 0 \
            --data_dir /data/mmlu/ \
            --save_dir {out_dir} \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --load_in_8bit \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "mmlu_5shot":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.mmlu.run_eval \
            --ntrain 5 \
            --data_dir /data/mmlu/ \
            --save_dir {out_dir} \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --load_in_8bit \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "bbh_direct":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.bbh.run_eval \
            --data_dir /data/bbh \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --max_num_examples_per_task 40 \
            --no_cot \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "bbh_cot":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.bbh.run_eval \
            --data_dir /data/bbh \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --max_num_examples_per_task 40 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "gsm_direct":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.gsm.run_eval \
            --data_dir /data/gsm/ \
            --max_num_examples 200 \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --n_shot 8 \
            --no_cot \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "gsm_cot":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.gsm.run_eval \
            --data_dir /data/gsm/ \
            --max_num_examples 200 \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --n_shot 8 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "tydiqa_goldp_1shot":
        d["tasks"][0]["arguments"][0] = f'''
            python -m eval.tydiqa.run_eval \
            --data_dir /data/tydiqa/ \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "tydiqa_no_context_1shot":
        d["tasks"][0]["arguments"][0] = f'''
            python -m eval.tydiqa.run_eval \
            --data_dir /data/tydiqa/ \
            --no_context \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "codex_eval_temp_0.1":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.1 \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model
        '''
    elif experiment_group == "codex_eval_temp_0.8":
        d['tasks'][0]['arguments'][0] = f'''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.8 \
            --save_dir {out_dir} \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model
        '''
    elif experiment_group == "truthfulqa":
        d['tasks'][0]['arguments'][0] = f'''
        python -m eval.truthfulqa.run_eval \
            --data_dir /data/truthfulqa \
            --save_dir {out_dir} \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --metrics judge info mc \
            --preset qa \
            --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
            --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
            --eval_batch_size 20 \
            --load_in_8bit \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "toxigen":
        d['tasks'][0]['arguments'][0] = f'''
        python -m eval.toxigen.run_eval \
            --data_dir /data/toxigen/ \
            --save_dir {out_dir} \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 32 \
            --use_vllm \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "alpaca_eval":
        d['tasks'][0]['arguments'][0] = f'''
        python -m eval.alpaca_farm.run_eval \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --save_dir {out_dir} \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    else:
        raise ValueError("experiment_group not supported")

    if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
    if model_info[1].startswith("/"):  # if it's a local model, load it from the local directory
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
    else:  # if it's a beaker model, mount the beaker dataset to `/model`
        d['tasks'][0]['datasets'][1]['source']['beaker'] = model_info[1]

    # if a specific checkpoint is specified, load model from that checkpoint
    if model_info[2] is not None:
        # extract existing model path
        model_name_or_path = re.search("--model_name_or_path (\S+)", d['tasks'][0]['arguments'][0]).group(1)
        # replace the model path with the checkpoint subfolder
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(model_name_or_path, model_name_or_path+"/"+model_info[2])]
        # replace the tokenizer path with the checkpoint subfolder
        tokenizer_name_or_path = re.search("--tokenizer_name_or_path (\S+)", d['tasks'][0]['arguments'][0]).group(1)

    # for vanilla_lm, remove the chat formatting function
    if model_info[3] == "vanilla_lm":
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]

    if "13B" in model_info[0]:
        # find the batch size argument, and reduce by 4x
        if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
            original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
            new_batch_size = max(1, int(original_batch_size) // 2)
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]


    if "30B" in model_info[0] or "34B" in model_info[0]:
        # find the batch size argument, and reduce by 4x
        if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
            original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
            new_batch_size = max(1, int(original_batch_size) // 4)
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

        if "codex_eval" in experiment_group:
            # request 2x more GPUs
            d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']

    elif "70B" in model_info[0] or "65B" in model_info[0] or "40B" in model_info[0]:
        # find the batch size argument, and reduce by 4x
        if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
            original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
            new_batch_size = max(1, int(original_batch_size) // 4)
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

        if "codex_eval" in experiment_group:
            # request 4x more GPUs
            d['tasks'][0]['resources']['gpuCount'] = 4 * d['tasks'][0]['resources']['gpuCount']
        else:
            # request 2x more GPUs
            d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']


    if "llama2-chat" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
            "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
        ]
    elif "code_llama_instruct" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
            "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
        ]
    elif "zephyr" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
            "--chat_formatting_function eval.templates.create_prompt_with_zephyr_chat_format")
        ]
    elif "xwin" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
            "--chat_formatting_function eval.templates.create_prompt_with_xwin_chat_format")
        ]

    if any([x in model_info[0] for x in ["opt", "pythia", "falcon"]]):
        if "--use_vllm" in d['tasks'][0]['arguments'][0]:
            print(f"Removing --use_vllm for {model_info[0]}")
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_vllm", "")]

    # print(d)

    fn = "beaker_configs/auto_created/{}.yaml".format(name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/science-adapt".format(fn)
    subprocess.Popen(cmd, shell=True)
