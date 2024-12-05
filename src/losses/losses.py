from functools import partial
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.evaluation.metrics import weighted_mean
from src.utilities.utils import get_logger


log = get_logger(__name__)


class LpLoss(torch.nn.Module):
    def __init__(
        self,
        p=2,
        relative: bool = True,
        weights: Optional[Tensor] = None,
        weighted_dims: Union[int, Iterable[int]] = (),
    ):
        """
        Args:
            p: Lp-norm type. For example, p=1 for L1-norm, p=2 for L2-norm.
            relative: If True, compute the relative Lp-norm, i.e. ||x - y||_p / ||y||_p.
        """
        super(LpLoss, self).__init__()

        if p <= 0:
            raise ValueError("Lp-norm type should be positive")

        self.p = p
        self.loss_func = self.rel if relative else self.abs
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        if weights is not None:
            self.mean_func = partial(weighted_mean, weights=weights)
        else:
            self.mean_func = torch.mean

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        # print(diff_norms.shape, y_norms.shape, self.mean_func)
        return self.mean_func(diff_norms / y_norms)

    def abs(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        return self.mean_func(diff_norms)

    def __call__(self, x, y):
        return self.loss_func(x, y)


def get_loss(name, reduction="mean", **kwargs):
    """Returns the loss function with the given name."""
    name = name.lower().strip().replace("-", "_")
    if name in ["l1", "mae", "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction, **kwargs)
    elif name in ["l2", "mse", "mean_squared_error"]:
        loss = nn.MSELoss(reduction=reduction, **kwargs)
    elif name in ["l2_rel"]:
        loss = LpLoss(p=2, relative=True, **kwargs)
    elif name in ["l1_rel"]:
        loss = LpLoss(p=1, relative=True, **kwargs)
    else:
        raise ValueError(f"Unknown loss function {name}")
    return loss
