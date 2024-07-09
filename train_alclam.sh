cd ../
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12346    train_alclam.py\
	--should_continue\
	--per_gpu_train_batch_size=$2
