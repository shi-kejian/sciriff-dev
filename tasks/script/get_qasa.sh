# Download the QASA dataset.
# Run this script from the `tasks` directory by invoking `bash script/get_qasa.sh`

# Exit if the dataset's already there.
if [ -e data/processed/qasa ]
then
    echo "QASA dataset already found downloaded; exiting."
    exit 0
fi

mkdir -p data/downloads/qasa
wget -O data/downloads/qasa/qasa_test_v1.1.json https://raw.githubusercontent.com/lgresearch/QASA/main/data/testset_answerable_1554_v1.1.json
mkdir -p data/processed/qasa
python script/process_qasa.py
rm -r "data/downloads/qasa"