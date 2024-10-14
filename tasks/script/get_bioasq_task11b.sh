# Process the bioasq_taskb dataset.
# Run this script from the `tasks` directory by invoking `bash script/get_bioasq_task11b.sh`

current_dir=$(pwd)

# Skip if already there.
if ls data/processed | grep bioasq > /dev/null
then
    echo "Processed bioasq_task11b dataset already found; exiting."
    exit 0
fi

mkdir -p data/downloads/bioasq_task11b
echo -en "BioASQ_task11b dataset is protected. Download the train and test split of the dataset from http://participants-area.bioasq.org/datasets/ and place it in data/downloads/bioasq_task11b. \nPress 'y' after downloading to continue, or any other key to exit ... "
read resume_again

if [ "$resume_again" = "y" ] || [ "$resume_again" = "Y" ]
then
    bioasq_train_file="data/downloads/bioasq_task11b/BioASQ-training11b.zip"
    bioasq_test_file="data/downloads/bioasq_task11b/Task11BGoldenEnriched.zip"
    if [ -f "$bioasq_train_file" ] && [ -f "$bioasq_test_file" ]
    then
        unzip -qj $bioasq_train_file -d "data/downloads/bioasq_task11b/train/"
        rm $bioasq_train_file
        unzip -qj $bioasq_test_file -d "data/downloads/bioasq_task11b/test/"
        rm $bioasq_test_file
    elif [ -d "data/downloads/bioasq_task11b/train" ] && [ -d "data/downloads/bioasq_task11b/test" ]
    then
        echo "Found files from previous download. Continuing ..."
    else
        echo "All required files not found. Exiting ..."
        exit 0
    fi
else
    echo "BioASQ_task11b data not processed. Exiting..."
    exit 0
fi

python script/process_bioasq_task11b.py
