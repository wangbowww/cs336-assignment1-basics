"""
    Optimizer Implementation
"""
from typing import Any, Dict

import torch
from torch import Tensor
from typing import Iterable, Dict, Any, Tuple, Optional, Callable

class AdamWOptim(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[Dict[str, Any]] | Iterable[Tuple[str, Tensor]],
        defaults: Dict[str, Any]
    ) -> None:
        # assuming we have lr, beta1, beta2, eps, lamda in defaults 
        super().__init__(params, defaults)
    
    def step(
        self,
        closure: Optional[Callable] = None
    ) -> None:
        
