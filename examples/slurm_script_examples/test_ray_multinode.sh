#!/bin/bash
#SBATCH --job-name=ray_multinode_test
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1GB
#SBATCH --ntasks-per-node=1
gpus_per_node=0
# load modules
module purge
conda init bash
source ~/.bashrc
# start conda
conda activate alpa_environment
# environment activated, check environment
python3 -V
python3 -c "from cupy.cuda import nccl"
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# start head node
port=6789
ip_head=$head_node_ip:$port
export ip_head

srun --nodes=1 --ntasks=1 -w "$head_node" \
	ray start --head --node-ip-address="$head_node_ip" --port=$port \
	--num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $gpus_per_node --block &

# start worker nodes
# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
	ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" \
	--num-gpus $gpus_per_node --block &
    sleep 5
done
# try ray
echo "test ray status"
ray list nodes --address "$ip_head"
ray list nodes
ray list actors
ray summary tasks
# end ray
ray stop
# exit environment
conda deactivate
exit
