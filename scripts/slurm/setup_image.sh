srun \
  --container-image=/netscratch/enroot/huggingface+transformers-pytorch-gpu+latest.sqsh \
  --container-workdir=/home/$USER/document_analysis_stack \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/document_analysis_stack:/home/$USER/document_analysis_stack \
  --container-save=/netscratch/$USER/pytorchlightning+transformers-pytorch-gpu+latest.sqsh \
  ./scripts/slurm/install.sh

