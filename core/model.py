"""
Neural network models for sensor-to-joint-angle prediction.

Implements:
  - HybridCNNLSTM: CNN for spatial features + LSTM for temporal modeling
  - SimpleCNN1D: Lightweight CNN-only baseline
"""

import numpy as np
import torch
import torch.nn as nn


def first_conv_importance(conv):
    """
    Sensor importance from the first convolution layer, which is the only layer
    that sees each sensor channel separately.

    The L1 norm of the weights attached to each input channel, normalised to sum
    to one. It shows how much the trained model draws on a channel, not a causal
    effect -- a guide for which channels to try removing next.
    """
    weights = conv.weight.cpu().detach().numpy()        # (out, in=sensors, k)
    importance = np.sum(np.abs(weights), axis=(0, 2))
    return importance / np.sum(importance)


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for shoulder joint angle prediction.

    Architecture:
      1. 1D CNN: spatial feature extraction across sensor channels
      2. LSTM: temporal dependency modeling
      3. Regression head: predicts 3 clinical joint angles
    """

    def __init__(self, num_sensors, seq_len, num_targets=3,
                 cnn_filters=32, lstm_hidden=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_sensors = num_sensors
        self.num_targets = num_targets

        self.cnn = nn.Sequential(
            nn.Conv1d(num_sensors, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_targets),
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, Num_Sensors)
        Returns:
            (Batch, num_targets) predicted joint angles
        """
        x_cnn = x.permute(0, 2, 1)            # (B, C, L)
        features = self.cnn(x_cnn)             # (B, cnn*2, L)
        features = features.permute(0, 2, 1)   # (B, L, cnn*2)
        _, (h_n, _) = self.lstm(features)      # h_n: (1, B, H)
        final_hidden = h_n.squeeze(0)          # (B, H)
        return self.regressor(final_hidden)    # (B, num_targets)

    def get_feature_importance(self):
        """
        Sensor importance based on L1-norm of first CNN layer weights.
        Returns normalized importance scores of shape (num_sensors,).
        """
        return first_conv_importance(self.cnn[0])


class SimpleCNN1D(nn.Module):
    """Simplified 1D CNN baseline (no LSTM)."""

    def __init__(self, num_sensors, seq_len, num_targets=3,
                 cnn_filters=32, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_sensors, cnn_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(cnn_filters * 2, num_targets)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x).squeeze(-1)
        return self.fc(x)

    def get_feature_importance(self):
        """Same measure as HybridCNNLSTM, so the Results tab can show it for both."""
        return first_conv_importance(self.cnn[0])


def create_model(config, num_sensors):
    """Factory function to create model from config dict."""
    model_type = config.get('model_type', 'hybrid_cnn_lstm')

    if model_type == 'hybrid_cnn_lstm':
        return HybridCNNLSTM(
            num_sensors=num_sensors,
            seq_len=config['sequence_length'],
            num_targets=len(config['target_cols']),
            cnn_filters=config['cnn_filters'],
            lstm_hidden=config['lstm_hidden'],
            dropout=config['dropout'],
        )
    elif model_type == 'simple_cnn':
        return SimpleCNN1D(
            num_sensors=num_sensors,
            seq_len=config['sequence_length'],
            num_targets=len(config['target_cols']),
            cnn_filters=config['cnn_filters'],
            dropout=config['dropout'],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
