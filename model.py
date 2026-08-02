import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig

MODEL_NAME = "CBSkyNet"
ALL_MODEL_TYPES = [MODEL_NAME]


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.dropout(x)


class CBAMChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = self.fc(x.mean(dim=[2, 3]))
        maximum = self.fc(x.amax(dim=[2, 3]))
        weights = torch.sigmoid(average + maximum).view(
            x.size(0), x.size(1), 1, 1
        )
        return x * weights


class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([average, maximum], dim=1)))
        return x * weights


class CBAMBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 11,
    ) -> None:
        super().__init__()
        self.spatial_attn = CBAMSpatialAttention(spatial_kernel)
        self.channel_attn = CBAMChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        return self.spatial_attn(x)


class CBSkyNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        in_channels: int = 1,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_input_dim: int = 256,
    ) -> None:
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels, 64, dropout)
        self.conv_block2 = ConvBlock(64, 128, dropout)
        self.conv_block3 = ConvBlock(128, 256, dropout)
        self.conv_block4 = ConvBlock(256, 512, dropout)
        self.pool = nn.AvgPool2d(2, 2)

        self.cbam4 = CBAMBlock(512, spatial_kernel=11)

        self.freq_proj = nn.Sequential(
            nn.Linear(512 * 8, lstm_input_dim),
            nn.LayerNorm(lstm_input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        bilstm_output_dim = lstm_hidden * 2
        self.lstm_norm = nn.LayerNorm(bilstm_output_dim)

        self.attn_pool = nn.Sequential(
            nn.Linear(bilstm_output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.fc1 = nn.Linear(bilstm_output_dim, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

        self.apply(init_weights)
        self._init_lstm()

    def _init_lstm(self) -> None:
        for name, parameter in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                hidden_size = self.bilstm.hidden_size
                parameter.data[hidden_size : 2 * hidden_size].fill_(1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool(self.conv_block1(x))
        x = self.pool(self.conv_block2(x))
        x = self.pool(self.conv_block3(x))
        x = self.cbam4(self.conv_block4(x))

        batch_size, channels, frequency_bins, time_steps = x.shape
        x = (
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size, time_steps, channels * frequency_bins)
        )
        x = self.freq_proj(x)

        lstm_output, _ = self.bilstm(x)
        lstm_output = self.lstm_norm(lstm_output)

        attention_weights = torch.softmax(self.attn_pool(lstm_output), dim=1)
        context = (lstm_output * attention_weights).sum(dim=1)

        embedding = F.relu(self.bn_fc(self.fc1(context)))
        embedding = self.dropout_layer(embedding)

        return {
            "logits": self.fc2(embedding),
            "embedding": embedding,
            "features": context,
        }


def get_model(
    config: ExperimentConfig,
    model_type: str = MODEL_NAME,
) -> nn.Module:
    if model_type != MODEL_NAME:
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. Use {MODEL_NAME!r}."
        )

    model_config = config.model
    return CBSkyNet(
        num_classes=model_config.num_classes,
        embedding_dim=model_config.embedding_dim,
        dropout=model_config.dropout,
    )

if __name__ == "__main__":
    from config import get_config

    experiment_config = get_config("pcen")
    model = get_model(experiment_config)
    dummy_input = torch.randn(4, 1, 64, 501)
    output = model(dummy_input)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("logits:", output["logits"].shape)
    print("parameters:", f"{parameter_count:,}")
