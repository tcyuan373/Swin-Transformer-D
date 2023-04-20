#!/bin/bash
# cd /home/ty373/workspace/transf_projects/dense_deit
# source /share/apps/anaconda3/2021.11/bin/activate
# conda init
# conda activate swin
# MODEL="dense_deit_Q_Res3_uniform_1_relu_tiny_patch16_224"
MASTERPORT=28201

python -m torch.distributed.launch --nproc_per_node 4 --master_port ${MASTERPORT} main.py \
--cfg configs/swinv2/swinv2_tiny_patch4_window8_256.yaml  --batch-size 4 \
--accumulation-steps 8