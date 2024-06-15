import torch
from torch import Tensor
from torch.nn import (Dropout, LayerNorm, Linear, Module, ReLU,
                      TransformerEncoder, TransformerEncoderLayer)

from domain.models.embedding.positional_encoding import PositionalEncoding


class TransformerPredictionModel(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dropout: float,
        num_features: int,
    ):
        super().__init__()
        self.hyperparameters = {
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dropout": dropout,
        }
        self.num_features = num_features
        self.pos_encoder = PositionalEncoding(num_features)
        self.input_linear = Linear(num_features, d_model)
        self.relu = ReLU()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.decoder = Linear(d_model, num_features)

    def forward(self, src: Tensor) -> Tensor:
        src = self.pos_encoder(src)
        src = self.input_linear(src)
        src = self.relu(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

    def get_hyperparameters_str(self):
        return "\n".join([f"{k}: {v}" for k, v in self.hyperparameters.items()])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictionModel(
        d_model=40,
        nhead=4,
        num_encoder_layers=6,
        dropout=0.1,
        num_features=4,
    ).to(device)

    with torch.no_grad():
        model.eval()
        fake_src = torch.rand(64, 30, 4).to(device)

        output = model(fake_src)
        print(output.shape)
