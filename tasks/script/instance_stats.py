"""
Analyze metadata distributions and token statistics of dataset instances in the ~/tasks/instances directory using multiprocessing.
It processes JSONL files in parallel, counts occurrences of various metadata attributes, computes token-related metrics,
and generates histograms and bar plots for the distribution of tasks, domains, input/output types, and token counts.
Results are saved in the specified 'results/statistics' directory. It supports token counting for 'llama' and 'gpt-4' (via `tiktoken`) models.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
import concurrent.futures

instances_path = Path("../instances")
results_path = Path("../results/statistics")
results_path.mkdir(parents=True, exist_ok=True)

task_counter = Counter()
domain_counter = Counter()
source_type_counter = Counter()
input_type_counter = Counter()
output_type_counter = Counter()

dataset_token_stats = defaultdict(lambda: {'total_tokens': 0, 'num_instances': 0, 'max_tokens': 0})

dataset_instance_counts = defaultdict(lambda: {'train': 0, 'validation': 0, 'test': 0})

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def count_tokens(text, tokenizer_type="gpt-4"):
    if tokenizer_type == "llama":
        num_tokens = len(tokenizer(text)['input_ids'])
    else:
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(text))
    return num_tokens

def plot_and_save(data, title, file_name):
    plt.figure(figsize=(18, 12))
    plot = sns.barplot(x=list(data.keys()), y=list(data.values()), hue=list(data.keys()), palette='viridis', legend=False)
    plot.set_title(title, fontsize=16)
    tick_labels = plot.get_xticklabels()
    locs = plot.get_xticks()
    # Set the x-tick locations using FixedLocator
    plot.xaxis.set_major_locator(plt.FixedLocator(locs))
    plot.set_xticklabels(tick_labels, rotation=90, fontsize=12)
    for index, value in enumerate(data.values()):
        plot.text(index, value, f'{value}', color='black', ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(results_path / file_name, dpi=300)
    plt.close()


def plot_and_save_hist(data, title, x_label, file_name):
    plt.figure(figsize=(14, 8))
    plot = sns.histplot(data, kde=True, log_scale=(True, False), color='skyblue')
    plot.set_title(title, fontsize=16)
    plot.set_xlabel(x_label, fontsize=14)
    plot.set_ylabel('Frequency', fontsize=14)
    for p in plot.patches:
        if p.get_height() > 0:
            plot.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                          textcoords='offset points')
    plt.tight_layout()
    plt.savefig(results_path / file_name, dpi=300)
    plt.close()

def dump_dataset_status_json(dataset_instance_counts, output_path):
    # At this point, all datasets should have all splits, so if a split has 0, warning that it's missing
    for dataset, counts in dataset_status.items():
        for split in ['train', 'validation', 'test']:
            if counts[split] == 0:
                print(f'WARNING: Dataset {dataset} is missing {split} instances.')
    # But for this script we allow it to fail silently
    dataset_status = {
        dataset: {
            'train': counts.get('train', 0), 
            'validation': counts.get('validation', 0),
            'test': counts.get('test', 0)
        } for dataset, counts in dataset_instance_counts.items()
    }
    
    with open(output_path / 'dataset_book.json', 'w') as out_file:
        # Format: {'dataset_name': {'train': num_train, 'validation': num_val, 'test': num_test}}
        json.dump(dataset_status, out_file, indent=4)

def plot_boxplot_token_counts_by_dataset(token_stats, file_name):
    data_to_plot = [(dataset, stats['total_tokens'] / stats['num_instances']) for dataset, stats in token_stats.items()]
    df_to_plot = pd.DataFrame(data_to_plot, columns=['Dataset', 'Average Token Count'])

    plt.figure(figsize=(20, 10))
    sns.boxplot(x='Dataset', y='Average Token Count', data=df_to_plot)
    plt.xticks(rotation=90)
    plt.title('Average Token Count by Dataset')
    plt.tight_layout()
    plt.savefig(results_path / file_name, dpi=300)
    plt.close()

def token_stats_single_process_init():
    # Ensure defaultdict is not shared across processes # noqa
    return {'total_tokens': 0, 'num_instances': 0, 'max_tokens': 0}

def process_files(dataset_path):
    local_task_counter = Counter()
    local_domain_counter = Counter()
    local_source_type_counter = Counter()
    local_input_type_counter = Counter()
    local_output_type_counter = Counter()
    local_token_stats = defaultdict(token_stats_single_process_init)
    local_instance_counts = defaultdict(int)

    for split_file in dataset_path.glob("*.jsonl"):
        split_name = split_file.stem
        with split_file.open('r', encoding='utf-8') as file:
            for line in file:
                instance = json.loads(line)
                metadata = instance.get("metadata", {})
                input_text = instance.get("input", "")
                output_text = instance.get("output", "")
                instance_token_count = count_tokens(input_text) + count_tokens(output_text)

                local_task_counter.update([metadata.get("task", "")])
                local_domain_counter.update(metadata.get("domains", []))
                local_source_type_counter.update([metadata.get("source_type", "")])
                local_input_type_counter.update([metadata.get("input_context", "")])
                local_output_type_counter.update([metadata.get("output_context", "")])
         
                stats = local_token_stats[dataset_path.name]
                stats['total_tokens'] += instance_token_count
                stats['num_instances'] += 1
                stats['max_tokens'] = max(stats['max_tokens'], instance_token_count)

                local_instance_counts[split_name] += 1

    return (local_task_counter, local_domain_counter, local_source_type_counter,
            local_input_type_counter, local_output_type_counter, dict(local_token_stats),
            dataset_path.name, dict(local_instance_counts))

# Merge local counters and stats into global ones
def merge_counters_and_stats(local_counters_and_stats):
    for (local_task_counter, local_domain_counter, local_source_type_counter, 
         local_input_type_counter, local_output_type_counter, local_token_stats,
         dataset_name, local_instance_counts) in local_counters_and_stats:

        task_counter.update(local_task_counter)
        domain_counter.update(local_domain_counter)
        source_type_counter.update(local_source_type_counter)
        input_type_counter.update(local_input_type_counter)
        output_type_counter.update(local_output_type_counter)

        for key, val in local_token_stats.items():
            dataset_token_stats[key]['total_tokens'] += val['total_tokens']
            dataset_token_stats[key]['num_instances'] += val['num_instances']
            dataset_token_stats[key]['max_tokens'] = max(dataset_token_stats[key]['max_tokens'], val['max_tokens'])

        for split, count in local_instance_counts.items():
            dataset_instance_counts[dataset_name][split] += count

def process_datasets():
    results = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_files, dataset_path): dataset_path for dataset_path in instances_path.iterdir() if dataset_path.is_dir()}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                dataset_path = futures[future]
                print(f'An exception occurred processing {dataset_path}: {exc}')
    # print("Results before merging:", results) # Debugging
    merge_counters_and_stats(results)

if __name__ == "__main__":
    process_datasets()

    data_for_dataframe = []
    for dataset, counts in dataset_instance_counts.items():
        for split in counts:
            instance_count = counts[split]
            data_tuple = (dataset, split, instance_count)
            data_for_dataframe.append(data_tuple)
    df_dataset_instance_counts = pd.DataFrame(
        data_for_dataframe,
        columns=['Dataset', 'Split', 'Instance Count']
    )

    # Print dataset instance counts
    print("================================ Dataset instance counts: ================================")
    print(tabulate(df_dataset_instance_counts, headers='keys', tablefmt='psql'))

    # Dump dataset status to JSON
    dump_dataset_status_json(dataset_instance_counts, Path('../tasks'))

    # Plot and save metadata distributions
    plot_and_save(task_counter, 'Task Distribution', 'task_distribution.png')
    plot_and_save(domain_counter, 'Domain Distribution', 'domain_distribution.png')
    plot_and_save(source_type_counter, 'Source Type Distribution', 'source_type_distribution.png')
    plot_and_save(input_type_counter, 'Input Type Distribution', 'input_type_distribution.png')
    plot_and_save(output_type_counter, 'Output Type Distribution', 'output_type_distribution.png')

    # Calculate average and max tokens per dataset
    average_tokens_per_dataset = {k: v['total_tokens'] / v['num_instances'] for k, v in dataset_token_stats.items()}
    max_tokens_per_dataset = {k: v['max_tokens'] for k, v in dataset_token_stats.items()}

    # Plot and save average and max token counts per dataset
    plot_and_save(average_tokens_per_dataset, 'Average Token Count per Dataset', 'average_token_count_per_dataset.png')
    plot_and_save(max_tokens_per_dataset, 'Max Token Count per Dataset', 'max_token_count_per_dataset.png')

    # Plot and save token histograms
    token_counts = [stat['total_tokens'] for stat in dataset_token_stats.values()]
    plot_and_save_hist(token_counts, 'Token Count Distribution', 'Token Count', 'token_count_distribution.png')

    # Plotting the distribution of dataset frequencies
    plt.figure(figsize=(52, 15))  # Width can be further increased to accommodate the large number of datasets
    sns.set(style="whitegrid")

    barplot = sns.barplot(
        x='Dataset',
        y='Instance Count',
        hue='Split',
        data=df_dataset_instance_counts,
        palette='viridis'
    )

    # Add the numbers on top of the bars
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:  # Only add text if there is a height (count) to display
            barplot.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                            textcoords='offset points')

    # Adjust the x-axis 
    plt.xticks(rotation=90)
    plt.title('Instance Counts per Dataset Split', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Instance Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(results_path / 'dataset_split_counts.png', dpi=300)
    plt.close()
    
    # Other stats ... 
    # E.g. Uncomment the following line to plot the boxplot
    # plot_boxplot_token_counts_by_dataset(dataset_token_stats, 'token_counts_by_dataset_boxplot.png')

