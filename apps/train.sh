#torchrun --nproc_per_node=2 --master_port=12345 train.py \
    # --model_name_or_path uer/gpt2-chinese-cluecorpussmall \
python train.py \
    --model_name_or_path fnlp/bart-base-chinese \
    --data_path /data/git-local/chaoslib/projects/genneral-ner/data/gner_train_test_split_0829/raw/train.jsonl \
    --val_data_path /data/git-local/chaoslib/projects/genneral-ner/data/gner_train_test_split_0829/raw/test.jsonl \
    --output_dir outputs \
    --per_device_train_batch_size 16 \
    --bf16 True \
    --num_train_epochs 50 \
    --save_steps 10000 \
    --learning_rate 2e-5 \
    --weight_decay 1e-1 \
    --save_total_limit=3 \
    --report_to tensorboard
