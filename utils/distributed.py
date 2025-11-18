"""
Distributed training utilities for multi-GPU setup.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Setup distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend for distributed training ('nccl', 'gloo')
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    print(f"Initialized process {rank}/{world_size}")


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def all_reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduce operation
    
    Returns:
        Reduced tensor
    """
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
    
    Returns:
        List of tensors from all processes
    """
    if not is_distributed():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    return tensor_list


def synchronize():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary of values across all processes.
    
    Args:
        input_dict: Dictionary of values to reduce
        average: Whether to average the values
    
    Returns:
        Reduced dictionary
    """
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    names = sorted(input_dict.keys())
    values = [input_dict[k] for k in names]
    
    # Convert to tensor
    values_tensor = torch.tensor(values, dtype=torch.float32, device='cuda')
    
    # All-reduce
    dist.all_reduce(values_tensor)
    
    # Average if requested
    if average:
        values_tensor /= world_size
    
    # Convert back to dict
    reduced_dict = {k: v.item() for k, v in zip(names, values_tensor)}
    
    return reduced_dict


if __name__ == "__main__":
    print("Distributed utilities module")