# Example usage: bash launch_compute_science_metrics.sh /net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/results
# Now that we have an LM judge, it's worth it to compute metrics in parallel and
# then aggregate at the end.
# This command computes the metrics in parallel. When it's finished, you can collect the results into a talbe with:
# python compute_science_metrics.py \
#     --pred_dir $pred_dir \
#     --metrics_dir $metrics_dir \
#     --baseline_model $baseline_model


result_dir=$1
pred_dir=$result_dir/predictions
metrics_dir=$result_dir/metrics

# Baseline model for LM judge comparisons.
baseline_model=llama_2_7b-tulu_all-science_none-seed_42_4096

ls $pred_dir | while read model_name
do
    mason \
        --cluster ai2/s2-cirrascale-l40 ai2/s2-cirrascale \
        --budget ai2/oe-adapt \
        --gpus 0 \
        --workspace ai2/science-adapt \
        --description "Science adapt metrics for ${model_name}." \
        --task_name "science_adapt_eval_${model_name}" \
        -- \
        python compute_science_metrics.py \
        --pred_dir $pred_dir \
        --metrics_dir $metrics_dir \
        --baseline_model $baseline_model \
        --model_name $model_name
done
