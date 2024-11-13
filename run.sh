export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=2
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
yein karne ki zarrorat nahi hai
11:18 AM
xport MASTER_ADDR=“192.168.10.3”
export MASTER_PORT=29603
export WORLD_SIZE=16
export NCCL_SOCKET_IFNAME=“ibp25s0"
export LOCAL_RANK=0
export RANK=1
###############
xport MASTER_ADDR=“192.168.10.3”
export MASTER_PORT=29603
export WORLD_SIZE=16
export NCCL_SOCKET_IFNAME=“ibp25s0"
export LOCAL_RANK=0
export RANK=8


torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 0 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite
torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 1 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite
torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 2 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite
torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 3 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite

torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 4 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite
torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 5 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite
torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 6 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite
torchrun --master_addr 10.67.54.1  --master_port 8000 --nnodes 8 --node_rank 7 --nproc_per_node 8 scripts/train.py configs/official/5TECHLAB.yaml --save_overwrite