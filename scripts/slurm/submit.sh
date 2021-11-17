#!/bin/bash -l

N_GPUS=
N_NODES=
PARTITION=
MEMORY=
CMD=

usage()
{
    echo "Usage:"
    echo "sbatch ./submit.sh --partition|-p=<partition> --memory|-m=<XG> --n_nodes=<n_nodes> --n_gpus=<n_gpus> --cmd=<cmd>"
    echo ""
    echo " -h | --help : Displays the help"
    echo " -p | --partition : GPU partition"
    echo " -m | --memory : Total memory"
    echo " --n_gpus : Number of GPUs per task."
    echo " --n_nodes : Number of nodes."
    echo " --cmd : Command to run. "
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
	    -p  | --partition )
	        PARTITION=$VALUE
	        ;;
	    -m  | --memory )
	        MEMORY=$VALUE
	        ;;
        --n_gpus )
            N_GPUS=$VALUE
            ;;
        --n_nodes)
            N_NODES=$VALUE
            ;;
        --cmd)
            CMD=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "$PARTITION" = "" ]; then
  usage
  exit 1
fi

if [ "$N_NODES" = "" ]; then
  usage
  exit 1
fi

if [ "$N_GPUS" = "" ]; then
  usage
  exit 1
fi

if [ "$MEMORY" = "" ]; then
  usage
  exit 1
fi

if [ "$CMD" = "" ]; then
  usage
  exit 1
fi

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=$N_NODES
#SBATCH --gres=gpu:$N_GPUS
#SBATCH --ntasks-per-node=$N_GPUS
#SBATCH --mem=$MEMORY
#SBATCH --partition $PARTITION

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# run script from above
srun \
    --container-image=/netscratch/$USER/pytorchlightning+transformers-pytorch-gpu+latest.sqsh \
    --container-workdir=/home/$USER/document_analysis_stack \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/document_analysis_stack:/home/$USER/document_analysis_stack \
    --task-prolog=./scripts/slurm/install.sh \
    -p $PARTITION \
    --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,NETSCRATCH_DAS=/netscratch/$USER/document_analysis_stack,PYTHONPATH=$PYTHONPATH:/home/$USER/document_analysis_stack/src,TORCH_HOME=/netscratch/$USER/document_analysis_stack/pretrained" \
    $CMD
