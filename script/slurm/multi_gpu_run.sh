srun -p V100-16GB --ntasks 8 --gpus-per-task 1 python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --data-path ./data  --output_dir /netscratch/saifullah/TWIST/output --aug docs --dim 16 --epochs 50

