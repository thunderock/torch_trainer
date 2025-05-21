#!/bin/bash
# set -e
set -x

# Default values
GIT_SHA=`git rev-parse --short=8 HEAD`
PLUTO_PROJECT="GPU-125-mldp-img-data-enrichment"
NUM_NODES=2
NUM_CPU_WORKERS_PER_NODE=8
NUM_GPU_WORKERS_PER_NODE=8
IMAGE=docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/colligo/colligo-dev:v42
OUT_DIR="/mnt/localssd/colligo"

# Parse command line arguments
while getopts "p:n:j:c:g:f:" opt; do
  case $opt in
    n)
      NUM_NODES=$OPTARG
      ;;
    j)
      JOB_NAME=$OPTARG
      ;;
    c)
      NUM_CPU_WORKERS_PER_NODE=$OPTARG
      ;;
    g)
      NUM_GPU_WORKERS_PER_NODE=$OPTARG
      ;;
    *)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

# Create a temporary wrapper script that will contain both the bootstrapping and job execution logic
WRAPPER_SCRIPT="wrapper.sh"
cat > "${WRAPPER_SCRIPT}" << EOF
#!/bin/bash
set -x
# Export variables for cluster configuration
export NUM_NODES=$NUM_NODES
export NUM_CPU_WORKERS_PER_NODE=$NUM_CPU_WORKERS_PER_NODE
export NUM_GPU_WORKERS_PER_NODE=$NUM_GPU_WORKERS_PER_NODE

echo NODE_RANK=\${RANK}

git clone git@github.com:thunderock/torch_trainer.git
cd torch_trainer
make setup

torchrun --nproc_per_node=8 --nnodes=${NUM_NODES} --node_rank=\${RANK} --master_addr=\${MASTER_ADDR} --master_port=29500 torch_trainer/model.py --num_nodes=${NUM_NODES} --gpus_per_node=${NUM_GPU_WORKERS_PER_NODE} --batch_size=32

EOF

# Make the wrapper script executable
chmod +x "${WRAPPER_SCRIPT}"

# Submit the job to Pluto
python3 -m colligo.pluto.sdk.cli job create \
    --job-type "training" \
    --name "${JOB_NAME}" \
    --image "${IMAGE}" \
    --project "${PLUTO_PROJECT}" \
    --accelerator-type NVIDIA_A100_80GB \
    --xpus-per-pod ${NUM_GPU_WORKERS_PER_NODE} \
    --num-pods ${NUM_NODES} \
    --start \
    --main-script "${WRAPPER_SCRIPT}"

# Clean up the wrapper script
rm "${WRAPPER_SCRIPT}"