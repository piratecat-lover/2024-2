import torch
import torch.nn as nn

from typing import Tuple


# use identity mlp when calculating q, k, v
class IdentityMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Linear(dim, dim)
        self.initialize_weights_ones()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def initialize_weights_ones(self):
        nn.init.ones_(self.mlp.weight)
        nn.init.ones_(self.mlp.bias)


# use HeaderConcatMLP when merging heads
class HeaderConcatMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        self.initialize_weights_ones()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def initialize_weights_ones(self):
        nn.init.ones_(self.mlp.weight)
        nn.init.ones_(self.mlp.bias)


def cal_window_transformer_block(
    x,
    #
    window_size: Tuple[int, int],
    num_heads: int,
):
    B, H, W, C = x.shape
    assert num_heads > 0, "num_heads must be greater than 0"
    assert (
        window_size[0] > 0 and window_size[1] < H and window_size[1] < W
    ), "window_size must be less than image size"

    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################

    ##############################################################################
    #                          END OF YOUR CODE                                  #
    ##############################################################################

    output: torch.Tensor = x
    B_, H_, W_, C_ = output.shape
    assert (B_, H_, W_, C_) == (
        B,
        H,
        W,
        C,
    ), "output shape should be same as input shape"
    return output
