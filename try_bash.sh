python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg configs/swinv2/swinv2_base_patch4_window8_256.yaml --data-path data/cifar-100-python --batch-size 64 \
--accumulation-steps 2