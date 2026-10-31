"""Small signature-compatible helpers for pinned upstream modules."""

from torch import nn


class ArgumentIdentity(nn.Identity):
    def forward(self, value, *args, **kwargs):
        return value


class BagAveragePool(nn.AdaptiveAvgPool1d):
    """Native adaptive mean pooling over tokens of a [B,N,D] bag."""

    def __init__(self):
        super().__init__(1)

    def forward(self, value):
        return super().forward(value.transpose(1, 2)).squeeze(-1)
