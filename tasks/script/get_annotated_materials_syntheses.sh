# Run this script from the `tasks` directory.

# Exit if the dataset's already there.
if [ -e data/processed/annotated_materials_syntheses ]
then
    echo "Materials synthesis dataset already found; exiting."
    exit 0
fi

# Download repo.
wget -P data/downloads https://github.com/olivettigroup/annotated-materials-syntheses/archive/refs/heads/master.zip
unzip data/downloads/master.zip -d data/downloads
rm data/downloads/master.zip

# Process the BRAT data.
python script/process_annotated_materials_syntheses.py
