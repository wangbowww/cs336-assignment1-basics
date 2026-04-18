"""
    tune the learning rate, using toy example
"""
from typing import Callable, Optional, Any
from torch import Tensor
import math
import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, defaults: dict[str, Any]) -> None:
        if "lr" not in defaults.keys() or defaults["lr"] < 0:
            raise ValueError(f"You must pass lr(>0) to construct Optimizer.")
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], {"lr": 100})
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

"""
{"lr": 10}      ✅️(decay but slow)
27.687341690063477
17.719898223876953
13.062353134155273
10.21989631652832
8.278116226196289
6.863506317138672
5.788457870483398
4.946402072906494
4.271606922149658
3.7210445404052734
"""

"""
{"lr": 100}     ✅️✅️(decay fast)
26.48383140563965
26.48382568359375
4.543905735015869
0.10874588042497635
1.3160479279921133e-16
1.4668172531588982e-18
4.9392893080101954e-20
2.9423688825047414e-21
2.52415395805746e-22
2.804615719315641e-23
"""

"""
{"lr": 1000}    ❌️(diverge)
22.315752029418945
8055.98486328125
1391395.375
154777904.0
12537009152.0
791228841984.0
40619103748096.0
1747607428792320.0
6.441308482699264e+16
2.0683757105268654e+18
"""