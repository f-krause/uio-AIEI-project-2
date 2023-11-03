import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from src.config import Config


class BaseModel(nn.Module):
    """Abstract class to inherit from"""

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.fc = nn.Linear(self.hidden_dim, config.output_dim)

    @abstractmethod
    def forward(self, x):
        """Needs to be defined in the child classes"""
        pass


class LSTMPred(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=config.input_dim, hidden_size=config.hidden_dim, num_layers=config.num_layers,
                            dropout=config.dropout, batch_first=True)

    def forward(self, x):
        # input should be (N, L, H_in) => (B, L, 1) since batch_first=True and nr_channels = 1
        _, (h_out, _) = self.lstm(x.unsqueeze(-1))
        out = self.fc(h_out.view(-1, self.hidden_dim))
        return out


class RNNPred(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.rnn = nn.RNN(input_size=config.input_dim, hidden_size=config.hidden_dim, num_layers=config.num_layers,
                          dropout=config.dropout, batch_first=True)

    def forward(self, x):
        # input should be (N, L, H_in) => (B, L, 1) since batch_first=True and nr_channels = 1
        _, h_out = self.rnn(x.unsqueeze(-1))
        out = self.fc(h_out.view(-1, self.hidden_dim))
        return out


class LSTMClass(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.lstm = nn.LSTM(input_size=config.input_dim, hidden_size=config.hidden_dim, num_layers=config.num_layers,
                            dropout=config.dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x.view(len(x), 1, -1))
        out_space = self.fc(out.view(len(x), -1))
        out_scores = F.log_softmax(out_space, dim=1)
        return out_scores


class RNNClass(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)
        self.rnn = nn.RNN(input_size=config.input_dim, hidden_size=config.hidden_dim, num_layers=config.num_layers,
                          dropout=config.dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x.view(len(x), 1, -1))
        out_space = self.fc(out.view(len(x), -1))
        out_scores = F.log_softmax(out_space, dim=1)
        return out_scores
