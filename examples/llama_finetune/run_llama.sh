export PYTHONPATH=$HOME/alpa/EasyLM:$PYTHONPATH
python3 run_easylm_flax.py \
    --output_dir="./output" \
    --model_name_or_path="/data/llama-7b" \
    --dataset_name="/data/sharegpt.json" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="16" \
    --num_micro_batches 32 \
    --operator_parallel 1 \
    --pipeline_parallel 8 \
    --dtype="float16" \
    --learning_rate="5e-4" --warmup_ratio="0.03" \
    --weight_decay="0.0" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --logging_steps="1" \
    --save_steps="3000" \
    --eval_steps="1000"
