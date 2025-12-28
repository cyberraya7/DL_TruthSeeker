import torch
import torch.nn as nn


class LuqTransformerModel(nn.Module):
    """
    Transformer-based architecture for tabular network intrusion data.

    Design:
    - Treat each input feature as a "token" in a sequence.
    - Project each scalar feature into a higher dimensional embedding (d_model).
    - Apply several Transformer encoder layers (self-attention + feedforward + LayerNorm).
    - Pool over the feature dimension and classify.

    This leverages self-attention to model interactions between features, which can be
    beneficial for complex tabular data like the CICIDS / truthseeker dataset.
    """

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_features = input_features

        # Project each scalar feature (shape [..., 1]) into a d_model-dimensional embedding.
        # We will reshape input from (B, F) -> (B, F, 1) before applying this.
        self.feature_proj = nn.Linear(1, d_model)

        # Transformer encoder layers operating over the "feature sequence".
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # input/output shape: (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_features)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Ensure expected shape
        # (B, F) -> (B, F, 1)
        x = x.unsqueeze(-1)

        # Project each scalar feature to an embedding: (B, F, 1) -> (B, F, d_model)
        x = self.feature_proj(x)

        # Self-attention over feature dimension
        x = self.encoder(x)
        x = self.norm(x)

        # Pool over features (sequence length) dimension: (B, F, d_model) -> (B, d_model)
        x = x.mean(dim=1)

        # Classification head
        logits = self.classifier(x)
        return logits


