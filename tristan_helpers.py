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
        self.data = data["value"].values

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
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add channel dimension
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the output of the last timestep
        return out.squeeze()
