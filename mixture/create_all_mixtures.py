from tqdm import tqdm
import itertools
from concurrent.futures import ProcessPoolExecutor

from lib import paths
from create_mixture import get_parser, create_mixture


def create_one_mixture(options):
    "Kick off creation of a single mixture."
    instance_root = paths.project_root / "data/instances"
    out_root = paths.project_root / "data/training_mixtures"
    context_window, n_insts, tulu, eval = options
    instance_dir = instance_root / str(context_window)
    out_dir = out_root / str(context_window)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create arguments to pass in to the `create_mixture` parser.
    parser = get_parser()
    arg_list = [
        "--instances_per_task",
        n_insts,
        "--instance_dir",
        instance_dir,
        "--tulu",
        tulu,
        "--out_dir",
        out_dir,
    ]
    if eval:
        arg_list.append("--include_eval_tasks")
    arg_list = [str(x) for x in arg_list]

    args = parser.parse_args(arg_list)
    create_mixture(args)


def make_small_tulu():
    "Make mixtures without the full Tulu dataset."
    # Do product of context windows, number of instances, and whether to include tulu.
    context_window_list = [4096, 8192]
    n_insts_list = [100, 200, 500, 1000]
    # Don't create versions with all tulu data for all settings; wastes too much storage.
    tulu_list = ["match", "none"]
    eval_list = [True, False]
    options_list = list(
        itertools.product(context_window_list, n_insts_list, tulu_list, eval_list)
    )

    print("Creating mixtures with less Tulu.")
    # Parallelize to speed things up a bit.
    with ProcessPoolExecutor(max_workers=32) as executor:
        executor.map(create_one_mixture, options_list)


def make_full_tulu():
    "Make mixtures with full Tulu dataset."
    # Do a couple versions with the full Tulu dataset; don't do all because it's
    # wasteful.
    context_window_list = [4096, 8192]
    n_insts_list = [0, 100, 200, 500, 1000]
    tulu_list = ["all"]
    eval_list = [False]

    options_list = list(
        itertools.product(context_window_list, n_insts_list, tulu_list, eval_list)
    )
    print("Creating mixtures with full Tulu.")
    with ProcessPoolExecutor(max_workers=32) as executor:
        executor.map(create_one_mixture, options_list)


def main():
    "Make a bunch of training mixtures."
    make_small_tulu()
    make_full_tulu()


if __name__ == "__main__":
    main()
