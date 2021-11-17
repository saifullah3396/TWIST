srun \
    --container-image=/netscratch/$USER/document_analysis_stack.sqsh \
    --container-workdir=/home/$USER/document_analysis_stack \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/document_analysis_stack:/home/$USER/document_analysis_stack,/home/$USER/TWIST:/home/$USER/TWIST \
    --task-prolog=./scripts/slurm/install.sh \
    -K \
    --ntasks=8 \
    --gpus-per-task=1 \
    --cpus-per-gpu=1 \
    --mem 30G \
    -p $PARTITION \
    --save=/netscratch/$USER/document_analysis_stack.sqsh \
    --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,NETSCRATCH_DAS=/netscratch/$USER/document_analysis_stack,PYTHONPATH=$PYTHONPATH:/home/$USER/document_analysis_stack/src,TORCH_HOME=/netscratch/$USER/document_analysis_stack/pretrained" \
    $CMD
    
srun \
    -p V100-16GB \
    --ntasks 8 \
    --gpus-per-task 1 \
    python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --data-path ./data  --output_dir /netscratch/saifullah/TWIST/output --aug docs --dim 16 --epochs 50

