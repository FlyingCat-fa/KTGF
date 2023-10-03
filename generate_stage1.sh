CUDA_VISIBLE_DEVICES=4 \
    python generate_stage1.py \
    --pretrained_model_cfg pretrained_models/t5-base-chinese-cluecorpussmall \
    --origin_data_dir dataset/Chunyu \
    --data_dir processed \
    --train_file train.json \
    --dev_file test.json \
    --output_dir dataset/Chunyu/exp_t5_base_chinese \
    --log_dir log \
    --model_recover_dir model.{}.bin \
    --dev_batch_size 256 \
    --add_category \
    --add_state \