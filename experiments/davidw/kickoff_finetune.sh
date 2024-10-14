# python run_finetune.py \
#     --train_file "mixtures_combined/4k_balance_task_80k.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale

# python run_finetune.py \
#     --train_file "mixtures_combined/tuluv2_no_science.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale

# python run_finetune.py \
#     --train_file "mixtures_combined/4k_balance_task_80k_with_eval.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale


# python run_finetune.py \
#     --train_file "mixtures_combined/4k_balance_task_40k.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale



####################

# Continued finetuning

# python run_finetune.py \
#     --train_file "mixtures_science_formatted/4k_per_task_100.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale \
#     --continued_finetune


# python run_finetune.py \
#     --train_file "mixtures_science_formatted/4k_per_task_200.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale \
#     --continued_finetune

# python run_finetune.py \
#     --train_file "mixtures_combined_tulu_frac/4k_per_task_1000_tulu_ratio_1.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale \
#     --continued_finetune

# python run_finetune.py \
#     --train_file "mixtures_combined/4k_balance_task_10k.jsonl" \
#     --num_gpus 8 \
#     --preprocessing_num_workers 60 \
#     --cluster ai2/allennlp-cirrascale


# python run_finetune.py \
#     --train_file "mixtures_combined/4k_per_task_500.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale


# python run_finetune.py \
#     --train_file "mixtures_combined/4k_per_task_1000.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale



# python run_finetune.py \
#     --train_file "mixtures_combined/4k_balance_task_10k.jsonl" \
#     --num_gpus 8 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/mosaic-cirrascale-a100


# python run_finetune.py \
#     --train_file "mixtures_combined/4k_balance_task_5k.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale


# python run_finetune.py \
#     --train_file "mixtures_combined/4k_per_task_200.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale




####################

# Failed; need to rerun.

# python run_finetune.py \
#     --train_file "mixtures_combined/4k_per_task_100.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale

# python run_finetune.py \
#     --train_file "mixtures_combined/4k_per_task_200.jsonl" \
#     --num_gpus 4 \
#     --preprocessing_num_workers 30 \
#     --cluster ai2/allennlp-cirrascale




####################

DATA_DIR=/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/training_mixtures/4096

# python run_finetune.py \
#     --train_file $DATA_DIR/tulu_all_science_100_eval_no.jsonl \
#     --beaker_model_path Yizhongw03/hf_llama2_model_7B \
#     --num_gpus 8 \
#     --preprocessing_num_workers 60 \
#     --cluster ai2/general-cirrascale-a100-80g-ib


# python run_finetune.py \
#     --train_file $DATA_DIR/tulu_all_science_500_eval_no.jsonl \
#     --beaker_model_path Yizhongw03/hf_llama2_model_7B \
#     --num_gpus 8 \
#     --preprocessing_num_workers 60 \
#     --cluster ai2/general-cirrascale-a100-80g-ib
