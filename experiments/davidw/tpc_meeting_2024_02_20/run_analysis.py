import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

from lib import paths, util

metrics = pd.read_table(paths.project_root / "_old/_old_eval_results/tables/metrics.tsv", index_col=0, header=[0, 1])


PLOT_COLS = [("science", "mean"), ("open_instruct", "mean")]


def plot_sample_scaling(metrics):
    rows = [
        "science_adapt_1-stage_tuluv2_no_science",
        "science_adapt_1-stage_4k_per_task_200",
        "science_adapt_1-stage_4k_per_task_500",
        "science_adapt_1-stage_4k_per_task_1000",
    ]
    to_plot = metrics.loc[rows][PLOT_COLS]
    to_plot.columns = ["science", "general"]
    to_plot.index = ["0", "200", "500", "1000"]


    # Create a plot with a secondary y-axis
    fig, ax = plt.subplots(1, figsize=[8, 4])

    to_plot.plot(ax=ax, marker=".")
    ax.set_ylim([0.15, 0.5])
    ax.set_title("Performance as a function of science training examples")
    ax.set_xlabel("Science examples per task")
    ax.set_ylabel("Aggregate performance")
    fig.tight_layout()
    fig.savefig("fig/data_scaling.png")


def plot_finetuning_strategy(metrics):
    rows = [
        "science_adapt_1-stage_tuluv2_no_science",
        "science_adapt_continued_4k_per_task_200",
        "science_adapt_continued_4k_per_task_200_tulu_ratio_1",
        "science_adapt_1-stage_4k_per_task_200",
    ]
    to_plot = metrics.loc[rows][PLOT_COLS]
    to_plot.index = ["No science", "Continued science only", "Continued science + general", "Standard finetuning"]
    to_plot.columns = ["science", "general"]

    # Create a plot with a secondary y-axis
    fig, ax = plt.subplots(1, figsize=[8, 4])

    to_plot.plot(ax=ax, marker="o", linestyle="", markersize=8)
    ax.set_ylim([0.15, 0.5])
    ax.set_title("Performance as a function of finetuning strategy")
    ax.set_xlabel("")
    ax.set_ylabel("Aggregate performance")
    ax.set_xticks(ticks=np.arange(len(to_plot.index)), labels=to_plot.index)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("fig/continued_finetune.png")


####################

rows = ["science_adapt_1-stage_4k_per_task_1000", "tulu-2-dpo-70b", "Llama-2-70b-chat-hf", "gpt-3.5-turbo-16k", "claude-2", "gpt-4"]
rows = rows[::-1]
to_plot = metrics.loc[rows][PLOT_COLS]
to_plot.columns = ["science", "general"]
to_plot = to_plot["science"]
to_plot.index = [
    "Science adapt 7B",
    "Tulu 2 70B DPO",
    "Llama 2 Chat 70B",
    "GPT-3.5",
    "Claude 2",
    "GPT-4",
][::-1]
colors = plt.cm.tab10.colors
fig, ax = plt.subplots(1, figsize=[8, 4])
ax.set_xlim(0.2, 0.6)
ax.set_title("Science performance of baseline models")
ax.set_xlabel("Aggregate science performance")
to_plot.plot.barh(ax=ax, color=colors)
fig.tight_layout()
fig.savefig("fig/baselines.png")


####################

# plot_sample_scaling()
