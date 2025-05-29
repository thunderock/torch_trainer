#!/bin/bash
# set -e
set -x

# Source .bashrc to get access to the submit_training_job function
source ~/.bashrc

# Default values
GIT_SHA=`git rev-parse --short=8 HEAD`
PLUTO_PROJECT="GPU-125-mldp-img-data-enrichment"
NUM_NODES=2
NUM_CPU_WORKERS_PER_NODE=8
NUM_GPU_WORKERS_PER_NODE=8
NUM_GPUS_PER_NODE=$NUM_GPU_WORKERS_PER_NODE  # Set this explicitly
JOB_NAME="torch_trainer"
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

# Set environment variables for distributed training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# Export variables for cluster configuration
export NUM_NODES=$NUM_NODES
export NUM_CPU_WORKERS_PER_NODE=$NUM_CPU_WORKERS_PER_NODE
export NUM_GPU_WORKERS_PER_NODE=$NUM_GPU_WORKERS_PER_NODE
export NUM_GPUS_PER_NODE=$NUM_GPU_WORKERS_PER_NODE

echo NODE_RANK=\${RANK}

cd ~
# how much space does it have?
df -h /
GIT_SSH_COMMAND="ssh -i /sensei-fs/users/astiwari/.ssh/id_rsa -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone git@github.com:thunderock/torch_trainer.git
cd torch_trainer

# Install required system libraries
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libbz2-dev libffi-dev liblzma-dev libncurses5-dev libreadline-dev libsqlite3-dev libtk8.6 libgdbm-dev uuid-dev libffi-dev

# Install and configure pyenv
curl https://pyenv.run | bash
export PYENV_ROOT="\$HOME/.pyenv"
[[ -d \$PYENV_ROOT/bin ]] && export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv init - bash)"
eval "\$(pyenv virtualenv-init -)"

# Now install and use Python
pyenv install 3.10.12
pyenv global 3.10.12

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/colligo/.local/bin:\$PATH"

make setup

poetry run torchrun --nproc_per_node=${NUM_GPU_WORKERS_PER_NODE} --nnodes=${NUM_NODES} --node_rank=\${RANK} --master_addr=\${MASTER_ADDR} --master_port=29500 torch_trainer/model.py --num_nodes=${NUM_NODES} --gpus_per_node=${NUM_GPU_WORKERS_PER_NODE} --batch_size=32 --max_epochs=10000

EOF

# Make the wrapper script executable
chmod +x "${WRAPPER_SCRIPT}"

# Submit the job to Pluto
submit_training_job "${WRAPPER_SCRIPT}" "${NUM_NODES}" "${JOB_NAME}" "${IMAGE}" "${PLUTO_PROJECT}" "${NUM_GPU_WORKERS_PER_NODE}"

# Clean up the wrapper script
rm "${WRAPPER_SCRIPT}"