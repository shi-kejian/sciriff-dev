
import datasets

from tasks.task import TaskCollection
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

def get_stats(task):
    "Get stats for a single task."
    metadata = deepcopy(task.metadata)
    res = {"name": task.name}
    metadata["domains"] = " | ".join(metadata["domains"])
    del metadata["contributor"]
    res.update(metadata)
    ds = datasets.load_dataset("ai2-adapt-dev/science-adapt-4096", task.name)
    counts = {"n_train": ds["train"].num_rows,
              "n_validation": ds["validation"].num_rows,
              "n_test": ds["test"].num_rows}

    # Just look at train token counts
    toks_input = ds["train"]["_input_toks"]
    toks_output = ds["train"]["_output_toks"]

    counts["total_toks_input"] = int(pd.Series(toks_input).sum())
    counts["total_toks_output"] = int(pd.Series(toks_output).sum())
    counts["total_toks"] = counts["total_toks_input"] + counts["total_toks_output"]
    counts["median_toks_input"] = int(pd.Series(toks_input).median())
    counts["median_toks_output"] = int(pd.Series(toks_output).median())
    counts["mean_toks_input"] = float(pd.Series(toks_input).mean())
    counts["mean_toks_output"] = float(pd.Series(toks_output).mean())
    res.update(counts)

    return res

# Make stats file, or load if already done.
stats_file = Path("results/stats.tsv")
if stats_file.exists():
    df = pd.read_table(stats_file)
else:
    tc = TaskCollection()
    workers = 20
    stats_file.parent.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = list(executor.map(get_stats, tc.tasks.values()))
        df = pd.DataFrame(res).set_index("name").sort_index()
        df.to_csv(stats_file, sep="\t")


# %%
import datasets

from tasks.task import TaskCollection
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

def get_stats(task):
    "Get stats for a single task."
    metadata = deepcopy(task.metadata)
    res = {"name": task.name}
    metadata["domains"] = " | ".join(metadata["domains"])
    del metadata["contributor"]
    res.update(metadata)
    ds = datasets.load_dataset("ai2-adapt-dev/science-adapt-4096", task.name)
    counts = {"n_train": ds["train"].num_rows,
              "n_validation": ds["validation"].num_rows,
              "n_test": ds["test"].num_rows}

    # Just look at train token counts
    toks_input = ds["train"]["_input_toks"]
    toks_output = ds["train"]["_output_toks"]

    counts["total_toks_input"] = int(pd.Series(toks_input).sum())
    counts["total_toks_output"] = int(pd.Series(toks_output).sum())
    counts["total_toks"] = counts["total_toks_input"] + counts["total_toks_output"]
    counts["median_toks_input"] = int(pd.Series(toks_input).median())
    counts["median_toks_output"] = int(pd.Series(toks_output).median())
    counts["mean_toks_input"] = float(pd.Series(toks_input).mean())
    counts["mean_toks_output"] = float(pd.Series(toks_output).mean())
    res.update(counts)

    return res

stats_file = Path("results/stats.tsv")

tc = TaskCollection()
stats_file.parent.mkdir(exist_ok=True)
results = [get_stats(task) for task in tc.tasks.values()]
df = pd.DataFrame(results).set_index("name").sort_index()
df.to_csv(stats_file, sep="\t")

# %%
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 12})

def make_pie_charts(df, field):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    task_counts = df[field].value_counts().sort_index()
    task_counts.plot.pie(ax=axs[0], autopct=lambda p: '{:.0f}'.format(p * sum(task_counts) / 100))
    axs[0].set_title("Number of tasks")

    inst_counts = df.groupby(field)["n_train"].sum().sort_index()
    inst_counts.plot.pie(ax=axs[1], autopct=lambda p: '{:.0f}'.format(p * sum(inst_counts) / 100))
    axs[1].set_title("Number of train instances")

    tok_counts = df.groupby(field)["total_toks"].sum().sort_index()
    print("Token counts:")
    print(tok_counts)
    # The total token counts can be really unbalanced; truncate the largest one so it's
    # the size of the rest combined.
    # Calculate the sum of all values except the largest one
    sum_except_max = tok_counts.sum() - tok_counts.max()
    # # Replace the largest value with this sum
    tok_counts.loc[tok_counts.idxmax()] = sum_except_max
    tok_counts.plot.pie(ax=axs[2], autopct=lambda p: '{:.2e}'.format(p * sum(tok_counts) / 100))
    axs[2].set_title("Number of train tokens (input + output).")

    axs[3].axis("off")

    fig.suptitle(field)
    fig.tight_layout()

# %% [markdown]
# ## Aggregate statistics
# 
# Total number of instances and tokens below.

# %%
df[
    [
        "n_train",
        "n_validation",
        "n_test",
        "total_toks_input",
        "total_toks_output",
        "total_toks",
    ]
].sum().apply(lambda x: f"{x:,}")

# %% [markdown]
# ## Task type statistics
# 
# Number of tasks per task type, and number of train instances per task type. There are lot of summarization instances because it's easy to get weakly-supervised data (e.g. use abstracts as summaries of papers, papers summaries on OpenReview, etc).
# 
# NOTE: token counts are very imbalanced (summarization has much more tokens than every other task), so in the plot I truncate the largest token count to be the sum of the token counts for the remaining gruops for easy of visualization.

# %%
from matplotlib import pyplot as plt

df["major_category"] = [x.split(".")[0] for x in df["task"]]
make_pie_charts(df, "major_category")


# %% [markdown]
# ## Domain statistics
# 
# Note that some tasks can be associated with multiple domains, so there's some double-counting here. It's fairly heavy on biomed, but there's a good amount of AI instances as well.
# 
# Based on this, maybe we should combine chemistry and materials science?

# %%
domain_df = []

for _, row in df.iterrows():
    domains = row["domains"].split(" | ")
    for domain in domains:
        domain_df.append({"domain": domain, "n_train": row["n_train"], "total_toks": row["total_toks"]})

domain_df = pd.DataFrame(domain_df)

make_pie_charts(domain_df, "domain")


# %%
# print mean token counts computed above
print(df[["mean_toks_input", "mean_toks_output"]].mean())

# %% [markdown]
# ## Input and output context
# 
# The types of input and output expected by the model. Interestingly, the majority of the tasks require `json` output. This happens for both IE tasks, as well as tasks like QA and entailment that require some kind of rationale justifying their answer.

# %%
make_pie_charts(df, "input_context")


# %%
make_pie_charts(df, "output_context")


# %% [markdown]
# ## Distribution of instances per task
# 
# Histograms showing the number of instances per task, across folds. The long tail at 10k is mostly summarization tasks for which we have essentially unlimited data.

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for split, ax in zip(["train", "validation", "test"], axs):
    df[f"n_{split}"].hist(ax=ax, bins=20)
    ax.set_xlabel("Instances")
    ax.set_ylabel("Count")
    ax.set_title(split)


# %% [markdown]
# ## Instance length distribution
# 
# Distribution of median input and output length, in tokens (using llama-2 tokenizer).

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for name, ax in zip(["input", "output"], axs):
    # Clip very long tasks.
    upper = 8096 if name == "input" else 1000
    df[f"median_toks_{name}"].clip(upper=upper).hist(ax=ax, bins=20)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Count")
    ax.set_title(name)

# Mean token

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for name, ax in zip(["input", "output"], axs):
    # Clip very long tasks.
    upper = 8096 if name == "input" else 1000
    df[f"mean_toks_{name}"].clip(upper=upper).hist(ax=ax, bins=20)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Count")
    ax.set_title(name)


# %% [markdown]
# ## Full list of tasks

# %%
fields_to_show = [ "domains", "task", "source_type", "input_context", "output_context", "n_train"]
df[fields_to_show].sort_values(["domains", "task"])

# %%
df

# %%
# For each task type (the key is "task"), get the mean token counts for input and output

df.groupby("task")[["mean_toks_input", "mean_toks_output"]].mean()

# %%
df2.head()

# %%



