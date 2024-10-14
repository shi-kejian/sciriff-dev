# Process the genia ataset.
# Run this script from the `tasks` directory by invoking `bash script/get_genia.sh`

# Exit if the dataset's already there.
if [ -e data/processed/genia ]
then
    echo "Processed genia dataset already found; exiting."
    exit 0
fi

tar -xvf data/lfs_data/genia.tar.gz -C data/downloads

mkdir -p data/processed/genia
python script/process_genia.py
