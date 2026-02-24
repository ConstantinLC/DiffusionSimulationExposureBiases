import torch.distributed as dist
import os
import torch

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank, global_rank

def cleanup():
    dist.destroy_process_group()