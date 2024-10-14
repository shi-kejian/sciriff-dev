"""
Upload all datasets to hub.
"""

import datasets
import argparse

from lib import paths


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to upload. Defaults to uploading all.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    tasks = args.tasks.split(",") if args.tasks else None

    for context_window_dir in paths.INSTANCE_DIR.iterdir():
        context_window = context_window_dir.name
        print(f"Uploading tasks for context window {context_window}.")
        for task_dir in context_window_dir.iterdir():
            task_name = task_dir.name

            # Skip it if not one of the specified tasks.
            if tasks is not None and task_name not in tasks:
                continue

            ds = datasets.load_dataset(
                "json", data_dir=task_dir, download_mode="force_redownload"
            )
            ds.push_to_hub(
                repo_id=f"ai2-adapt-dev/science-adapt-{context_window}",
                config_name=task_name,
                private=True,
            )


if __name__ == "__main__":
    main()
