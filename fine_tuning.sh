cd ../

HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1   python fine_tuning.py \
  --dataset $2 \
  --pretrained_bert_name $3

