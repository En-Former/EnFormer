# default seed: 42
TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 29511 \
train.py \
--model enformer_small \
--batch-size 256 \
--lr 0.001 \
--drop-path 0.1 \
--data_dir /data/imagenet1k \
--epochs 300 \
--experiment enformer_small_4xb256_1e-3_dp01_300e \
--checkpoint-hist 4 \
--pin-mem \
--amp \
#--resume "" \
