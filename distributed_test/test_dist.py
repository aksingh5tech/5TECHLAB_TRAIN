import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.1.1'  # Change to the appropriate IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    local_rank = rank % torch.cuda.device_count()  # Modulo the number of GPUs per node
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    # Model
    model = nn.Sequential(nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 10)).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('.', download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')

    cleanup()

if __name__ == "__main__":
    world_size = 16  # total number of GPUs across all nodes
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
