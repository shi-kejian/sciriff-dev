#! /usr/bin/bash

# Main script to build dataset.

python validate.py   # Validate schema.
bash script/get_all_datasets.sh

out_root=$PROJECT_ROOT/data/instances
mkdir -p $out_root

# Create versions with different context windows.
for context_window in 4096 8192 16384
do
    out_dir=$out_root/${context_window}
    python instantiate.py \
        --template 0 \
        --n_instances 2500 \
        --context_window $context_window \
        --workers 60 \
        --out_dir $out_dir

    # Write-protect the data just to be safe.
    chmod -R a-w $out_dir/*
done

# Some quick checks that everything looks ok.
pytest test_instances.py
