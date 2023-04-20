#!/bin/bash
# cd /home/ty373/workspace/transf_projects/dense_deit
# source /share/apps/anaconda3/2021.11/bin/activate
# conda init
# conda activate swin
MASTERPORT=28201


python -m torch.distributed.launch --nproc_per_node 4 --master_port ${MASTERPORT} main.py \
--cfg configs/swindense/swindense_res3.yaml  --batch-size 32 \
--accumulation-steps 2