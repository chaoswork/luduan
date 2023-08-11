

# Simple Test
# python train.py \
#     --data_path ./tiny_data/shakespeare.txt \
#     --output_dir outputs \
#     --per_device_train_batch_size 16

# Single GPU Test
CUDA_VISIBLE_DEVICES=1 python train.py \
    --data_path stas/openwebtext-10k \
    --data_type huggingface \
    --output_dir outputs.t \
    --per_device_train_batch_size 32 \
    --bf16 True \
    --max_steps 600 \
    --logging_steps 100 \
    --learning_rate 6e-4 \
    --weight_decay 1e-1 \
    --save_total_limit=3 \
    --mmap_file_path build/data/openwebtext-10k.bin \
    --report_to tensorboard \
    --gradient_accumulation_steps 4
    # --save_steps 100 \

exit 0
# Single GPU Test
CUDA_VISIBLE_DEVICES=1 python train.py \
    --data_path openwebtext \
    --data_type huggingface \
    --output_dir outputs \
    --per_device_train_batch_size 8192 \
    --bf16 True \
    --max_steps 600000 \
    --save_steps 10000 \
    --learning_rate 6e-4 \
    --weight_decay 1e-1 \
    --save_total_limit=3 \
    --mmap_file_path build/data/openwebtext.bin \
    --report_to tensorboard
    # --save_steps 100 \
