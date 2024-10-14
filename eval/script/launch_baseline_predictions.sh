# Make predictions for modeling baselines.

# Open models (Tulu and llama) available on Huggingface.

# The 70b models require 4 A600's or 2 A100's.


RESULT_DIR=/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/results/baseline-predictions

# Llama

python script/predict_eleuther.py \
    --model=vllm \
    --model_name=/llama_2_7b_chat \
    --beaker_dataset=Yizhongw03/hf_llama2_model_7B \
    --gpus 1 \
    --result_base $RESULT_DIR \
    --beaker_cluster "ai2/s2-cirrascale-l40" \
    --beaker_budget ai2/oe-adapt


python script/predict_eleuther.py \
    --model=vllm \
    --model_name=/llama_3_8b_instruct \
    --beaker_dataset=davidw/Meta-Llama-3-8B-Instruct \
    --gpus 1 \
    --result_base $RESULT_DIR \
    --beaker_cluster "ai2/s2-cirrascale-l40" \
    --beaker_budget ai2/oe-adapt


python script/predict_eleuther.py \
    --model=vllm \
    --model_name=/llama_2_70b_chat \
    --beaker_dataset=Yizhongw03/hf_llama2_model_70B \
    --gpus 4 \
    --result_base $RESULT_DIR \
    --beaker_cluster "ai2/allennlp-cirrascale" \
    --beaker_budget ai2/oe-adapt

python script/predict_eleuther.py \
    --model=vllm \
    --model_name=/llama_3_70b_instruct \
    --beaker_dataset=davidw/Meta-Llama-3-70B-Instruct \
    --gpus 4 \
    --result_base $RESULT_DIR \
    --beaker_cluster "ai2/allennlp-cirrascale" \
    --beaker_budget ai2/oe-adapt

####################

# Proprietary models

python script/predict_eleuther.py \
    --model=openai-chat-completions \
    --model_name=gpt-3.5-turbo-1106 \
    --result_base $RESULT_DIR \
    --beaker_budget ai2/oe-adapt

python script/predict_eleuther.py \
    --model=openai-chat-completions \
    --model_name=gpt-4-turbo-preview \
    --result_base $RESULT_DIR \
    --beaker_budget ai2/oe-adapt

# Claude 3 doesn't work with Eleuther; just use Claude 2 for now.
python script/predict_eleuther.py \
    --model=anthropic \
    --model_name=claude-2 \
    --result_base $RESULT_DIR \
    --beaker_budget ai2/oe-adapt
