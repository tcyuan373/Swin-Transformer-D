#!/bin/bash
# cd /home/ty373/workspace/transf_projects/dense_deit
# source /share/apps/anaconda3/2021.11/bin/activate
# conda init
# conda activate swin
# MODEL="dense_deit_Q_Res3_uniform_1_relu_tiny_patch16_224"

python -m torch.distributed.launch --nproc_per_node 1 --master_port 22803 main.py \
--cfg configs/swinv2/swinv2_tiny_patch4_window8_256.yaml  --batch-size 16 \
--accumulation-steps 0 --local_rank 0

# python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
# --cfg configs/swinv2/swinv2_tiny_patch4_window8_256.yaml --pretrained swinv2_tiny_patch4_window8_256.pth \
# --batch-size 16 --accumulation-steps 2 [--use-checkpoint] --local_rank 0