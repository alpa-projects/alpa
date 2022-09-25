python3 run_clm_flax.py \
    --output_dir="./output" \
    --model_name_or_path="facebook/opt-2.7b" \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-2-raw-v1" \
    --do_train --do_eval \
    --block_size="1024" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="32" \
    --num_micro_batches 64 \
    --operator_parallel 1 \
    --pipeline_parallel 2 \
    --dtype="float16" \
    --learning_rate="5e-4" --warmup_steps="2000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="8" \
    --logging_steps="4" \
    --save_steps="16" \
    --eval_steps="16"
