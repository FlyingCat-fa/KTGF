CUDA_VISIBLE_DEVICES=1 \
    python generate_stage1.py \
    --pretrained_model_cfg pretrained_models/t5-base-chinese-cluecorpussmall \
    --origin_data_dir dataset/CMDD \
    --data_dir processed \
    --train_file train.json \
    --dev_file test.json \
    --output_dir dataset/CMDD/exp_t5_base_chinese \
    --log_dir log \
    --model_recover_dir model.{}.bin \
    --dev_batch_size 256 \
    --add_state \
    --add_category \


CUDA_VISIBLE_DEVICES=1 \
    python generate_stage2.py \
    --pretrained_model_cfg pretrained_models/t5-base-chinese-cluecorpussmall \
    --origin_data_dir dataset/CMDD \
    --data_dir processed \
    --train_file train.json \
    --dev_file test.json \
    --output_dir dataset/CMDD/exp_t5_base_chinese \
    --log_dir log \
    --model_recover_dir model.{}.bin \
    --dev_batch_size 256 \
    --add_state \
    --add_category \

