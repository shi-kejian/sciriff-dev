"""
Run science evaluations for all available models.
"""

# NOTE: 7b model evals can run on a single l40 or A6000. 70b models require 8 A6000's or
# 4 A100's.


import predict_eleuther
from lib import paths


from beaker import Beaker


# For the earliest models I didn't specify "1-stage" vs. "continued"; map names.
name_lookup = {
    "science_adapt_4k_balance_task_40k": "science_adapt_1-stage_4k_balance_task_40k",
    "science_adapt_4k_balance_task_80k": "science_adapt_1-stage_4k_balance_task_80k",
    "science_adapt_4k_balance_task_80k_with_eval": "science_adapt_1-stage_4k_balance_task_80k_with_eval",
    "science_adapt_tuluv2_no_science": "science_adapt_1-stage_tuluv2_no_science",
}

b = Beaker.from_env(default_workspace="ai2/science-adapt")
workspace = b.workspace

experiments = workspace.experiments()

train_experiments = []
for exp in experiments:
    if "finetuning" in exp.description:
        train_experiments.append(exp)

merge_experiments = []
for exp in experiments:
    if "Merge models" in exp.description:
        merge_experiments.append(exp)

all_experiments = train_experiments + merge_experiments

result_dir = paths.project_root / "results" / "by_model"

for exp in all_experiments:
    if len(exp.jobs) != 1:
        continue

    name = exp.jobs[0].name
    status = exp.jobs[0].status.exit_code
    if status != 0:
        continue

    # Map old names.
    if name in name_lookup:
        name = name_lookup[name]

    model_name = f"/{name}"
    beaker_result = exp.jobs[0].result.beaker

    args = [
        "--model=vllm",
        f"--model_name={model_name}",
        "--tulu_format",
        "--gpus=1",
        "--cluster=ai2/s2-cirrascale",
        f"--beaker_dataset={beaker_result}",
        "--limit=500",
        "--budget=ai2/oe-adapt"
    ]

    parser = predict_eleuther.make_parser()
    args = parser.parse_args(args)
    predict_eleuther.kickoff(args)
