#!/bin/bash -l

N_GPUS=
N_NODES=
PARTITION=
MEMORY=
CMD=
SAVE=

usage()
{
    echo "Usage:"
    echo "./gpu_run.sh --partition|-p=<partition> --memory|-m=<XG> --n_nodes=<n_nodes> --n_gpus=<n_gpus> --cmd=<cmd> --save"
    echo ""
    echo " -h | --help : Displays the help"
    echo " -p | --partition : GPU partition"
    echo " -m | --memory : Total memory"
    echo " --n_gpus : Number of GPUs per task."
    echo " --n_nodes : Number of nodes."
    echo " --cmd : Command to run. "
    echo " --save: Whether to save container. "
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
        --save)
            SAVE=$VALUE
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

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


if [ "$SAVE" = "" ]; then
    srun \
        --container-image=/netscratch/$USER/document_analysis_stack.sqsh \
        --container-workdir=/home/$USER/document_analysis_stack \
        --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/document_analysis_stack:/home/$USER/document_analysis_stack,/home/$USER/TWIST:/home/$USER/TWIST \
        --task-prolog=./scripts/slurm/install.sh \
        -K \
        --ntasks=$N_GPUS \
        --gpus-per-task=$N_GPUS \
        --cpus-per-gpu=1 \
        --mem $MEMORY \
        -p $PARTITION \
        --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,NETSCRATCH_DAS=/netscratch/$USER/document_analysis_stack,PYTHONPATH=$PYTHONPATH:/home/$USER/document_analysis_stack/src,TORCH_HOME=/netscratch/$USER/document_analysis_stack/pretrained" \
        $CMD
else
    srun \
        --container-image=/netscratch/$USER/document_analysis_stack.sqsh \
        --container-workdir=/home/$USER/document_analysis_stack \
        --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/$USER/document_analysis_stack:/home/$USER/document_analysis_stack,/home/$USER/TWIST:/home/$USER/TWIST \
        --task-prolog=./scripts/slurm/install.sh \
        -K \
        --ntasks=$N_GPUS \
        --gpus-per-task=$N_GPUS \
        --cpus-per-gpu=1 \
        --mem $MEMORY \
        -p $PARTITION \
        --save=/netscratch/$USER/document_analysis_stack.sqsh \
        --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,NETSCRATCH_DAS=/netscratch/$USER/document_analysis_stack,PYTHONPATH=$PYTHONPATH:/home/$USER/document_analysis_stack/src,TORCH_HOME=/netscratch/$USER/document_analysis_stack/pretrained" \
        $CMD
fi
