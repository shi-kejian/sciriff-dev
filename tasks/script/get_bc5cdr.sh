# Process the BC5CDR datase since HF download links are not working anymore
# Run this script from the `tasks` directory by invoking `bash script/get_genia.sh`

# Exit if the dataset's already there.
if [ -e data/processed/bc5cdr ]
then
    echo "Processed BC5CDR dataset already found; exiting."
    exit 0
fi

unzip tasks/data/lfs_data/CDR_Data.zip -d tasks/data/lfs_data/