srun \
    --container-image=/netscratch/$USER/pytorchlightning+transformers-pytorch-gpu+latest.sqsh \
    --container-workdir=/home/$USER/document_analysis_stack \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/document_analysis_stack:/home/$USER/document_analysis_stack \
    --task-prolog=./scripts/slurm/install.sh \
    -K \
    --ntasks=1 \
    --gpus-per-task=1 \
    --cpus-per-gpu=1 \
    -p V100-32GB \
    --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,NETSCRATCH_DAS=/netscratch/$USER/document_analysis_stack,PYTHONPATH=$PYTHONPATH:/home/$USER/document_analysis_stack/src,TORCH_HOME=/netscratch/$USER/document_analysis_stack/pretrained" \
    $@
