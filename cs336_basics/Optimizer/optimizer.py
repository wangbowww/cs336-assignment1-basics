"""
    Optimizer Implementation
"""
from typing import Any, Dict

import torch
from torch import Tensor
from typing import Iterable, Dict, Any, Tuple, Optional, Callable
import math

class AdamWOptim(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[Dict[str, Any]] | Iterable[Tuple[str, Tensor]],
        lr: float,
        weight_decay: float,
        betas: tuple[float],
        eps: float,
    ) -> None:
        # assuming we have lr, beta1, beta2, eps, lamda in defaults 
        defaults = {}
        defaults["lr"] = lr
        defaults["weight_decay"] = weight_decay
        defaults["betas"] = betas
        defaults["eps"] = eps
        super().__init__(params, defaults)
    
    def step(
        self,
        closure: Optional[Callable] = None
    ) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            b1, b2 = group["betas"][0], group["betas"][1]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                state["m"] = b1 * m + (1 - b1) * grad
                state["v"] = b2 * v + (1 - b2) * grad * grad
                m = state["m"]
                v = state["v"]
                lrt = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                p.data -= lrt * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
        return loss
