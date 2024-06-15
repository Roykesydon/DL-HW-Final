import torch
from mamba_ssm import Mamba
from torch import Tensor
from torch.nn import (Dropout, LayerNorm, Linear, Module, ReLU,
                      TransformerEncoder, TransformerEncoderLayer)

from domain.models.embedding.positional_encoding import PositionalEncoding


class MambaPredictionModel(Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: float,
        num_features: int,
        input_days: int,
        output_days: int = 1,
    ):
        super().__init__()
        self.hyperparameters = {
            "d_model": d_model,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }
        self.num_features = num_features
        self.input_days = input_days
        self.output_days = output_days
        # self.pos_encoder = PositionalEncoding(num_features)
        self.input_linear = Linear(num_features, d_model)
        self.relu = ReLU()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_linear = Linear(d_model, num_features * output_days)

    def forward(self, src: Tensor) -> Tensor:
        src = self.input_linear(src)
        src = self.relu(src)
        output = self.mamba(src)
        output = self.output_linear(output)
        return output.view(-1, self.input_days, self.output_days, self.num_features)

    def get_hyperparameters_str(self):
        return "\n".join([f"{k}: {v}" for k, v in self.hyperparameters.items()])
