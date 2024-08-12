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


def l2_regularization(
        coeff: float,
        model: torch.nn.Module
) -> float:
    r"""
    Performs :math:`\ell_2` regularization on a PyTorch neural network.

    The regularization term of the total loss is given by

    .. math::

        \mathcal{L}_{\ell_2} := \lambda \sum_\theta \|\theta\|_2^2,

    where :math:`\lambda` is the regularization coefficient `coeff` and :math:`\theta` are the
    trainable parameters of the model.

    Args:
        coeff: Weighting coefficient of the regularization loss
        model: A `torch` neural network module with trainable parameters

    Returns:
        Scalar value of regularization loss
    """
    assert isinstance(model, torch.nn.Module), 'Model must be a torch.nn.Module object'
    
    params_norm = [(p ** 2).sum() for (_, p) in model.named_parameters()]
    return coeff * sum(params_norm)

