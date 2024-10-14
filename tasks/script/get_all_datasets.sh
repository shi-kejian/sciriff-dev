# Download all freely-available datasets.

mkdir -p data/downloads
mkdir -p data/processed

bash script/get_scierc.sh
bash script/get_chemprot.sh
bash script/get_genia.sh
bash script/get_qasa.sh
bash script/get_evidence_inference.sh
bash script/get_annotated_materials_syntheses.sh
bash script/get_bioasq_task11b.sh
