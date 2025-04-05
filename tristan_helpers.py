import pandas as pd
import jax.numpy as jnp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=24):
        self.seq_len = seq_len
        # self.data = data["value"].values
        self.data = data


    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # Input sequence
        y = self.data[idx + self.seq_len]  # Target value
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

class LSTMForecast(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x = x.unsqueeze(-1)  # Add channel dimension
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the output of the last timestep
        # return self.fc(out) # FOR MULTI-STEP PREDICTION!
        return out


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=2, dropout=0.1, seq_len=24):
        super(TimeSeriesTransformer, self).__init__()

        self.model_dim = model_dim
        self.seq_len = seq_len

        # 1. Input projection
        self.input_embedding = nn.Linear(input_dim, model_dim)

        # 2. Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim))

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output projection
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, x):
        """
        x: shape [batch_size, seq_len, input_dim]
        """
        # Project inputs to model dimension
        x = self.input_embedding(x)

        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Pass through Transformer
        out = self.transformer_encoder(x)

        # Use last timestep for prediction
        last_out = out[:, -1, :]  # shape: [batch_size, model_dim]

        # Final output
        return self.fc_out(last_out)







### TCN (Temporal Convolutional Network) for Time Series Forecasting
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Trim the padding to keep the output length equal to input length."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """A single residual block of the TCN."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of temporal blocks."""
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNForecast(nn.Module):
    """Full forecasting model using TCN + final dense layer."""
    def __init__(self, input_dim=1, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2):
        super(TCNForecast, self).__init__()
        self.tcn = TemporalConvNet(input_size=input_dim,
                                   num_channels=num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim] → [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        y = self.tcn(x)  # [batch, channels, seq_len]
        y = self.linear(y[:, :, -1])  # Use last output for prediction
        return y