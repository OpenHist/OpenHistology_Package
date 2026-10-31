"""Original, feature-level statistical pooling baselines."""

import torch
from torch import nn

from ..base import MILAdapter, as_bag, output_from_logits


class StatisticalMIL(MILAdapter):
    def __init__(self, name, input_dim, num_classes):
        super().__init__()
        self.name, self.input_dim = name, input_dim
        self.classifier = nn.Linear(input_dim * (2 if name == "mean_std_mil" else 1), num_classes)
        self.metadata = {
            "implementation": "original baseline",
            "pooling": name,
            "std_correction": 0 if name == "mean_std_mil" else None,
        }

    def forward(self, inputs, **native_kwargs):
        if native_kwargs:
            raise ValueError("Statistical pooling takes feature tensors only")
        values = as_bag(inputs, self.input_dim)
        if self.name == "mean_mil":
            embedding = values.mean(0, keepdim=True)
        elif self.name == "max_mil":
            embedding = values.max(0, keepdim=True).values
        else:
            embedding = torch.cat((values.mean(0), values.std(0, correction=0)), 0).unsqueeze(0)
        logits = self.classifier(embedding)
        return output_from_logits(logits, embedding=embedding, native=logits)


def build(name, input_dim=None, num_classes=None, **options):
    if options:
        raise ValueError("Statistical pooling does not accept additional model options")
    return StatisticalMIL(name, input_dim or 1024, num_classes or 2)
