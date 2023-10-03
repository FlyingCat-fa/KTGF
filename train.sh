CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
    --master_port 19604 \
    --nproc_per_node=1 \
    train.py \
    --pretrained_model_cfg pretrained_models/t5-base-chinese-cluecorpussmall \
    --data_dir dataset/Chunyu/processed \
    --train_file train.json \
    --dev_file dev.json \
    --output_dir dataset/Chunyu/exp_t5_base_chinese \
    --log_dir log \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --add_category \
    --add_state