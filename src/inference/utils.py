import math
import torch
import torch.optim as optim


def adjust_learning_rate(
    max_steps: int,
    warmup_fraction: float,
    base_lr: float,
    optimizer: optim.Optimizer,
    step: int
) -> float:
    warmup_steps = warmup_fraction * max_steps
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = 0
        lr = base_lr * q + end_lr * (1 - q)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
