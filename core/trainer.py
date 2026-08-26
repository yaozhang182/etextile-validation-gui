"""
Model training and evaluation module.

Handles training loop, LR scheduling, checkpointing, and inference.
Supports an optional progress callback for GUI integration.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def train_model(model, train_loader, config, device, output_dir=None,
                progress_callback=None):
    """
    Train the model.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        config: dict with 'epochs', 'learning_rate', 'batch_size'
        device: 'cuda' or 'cpu'
        output_dir: directory to save best checkpoint (optional)
        progress_callback: callable(epoch, epochs, loss) for GUI updates (optional)

    Returns:
        (trained_model, train_losses)
    """
    epochs = config['epochs']
    lr = config['learning_rate']

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    model.train()
    train_losses = []
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            if output_dir is not None:
                checkpoint_path = Path(output_dir) / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)

        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_loss)

    return model, train_losses


def evaluate_model(model, test_loader, device):
    """
    Run inference on test set.

    Returns:
        (predictions, ground_truth) as numpy arrays of shape (N, 3)
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_pred = model(x)
            all_preds.append(y_pred.cpu())
            all_true.append(y)

    predictions = torch.cat(all_preds, dim=0).numpy()
    ground_truth = torch.cat(all_true, dim=0).numpy()
    return predictions, ground_truth


def predict_single_sample(model, sensor_sequence, device, scaler=None):
    """
    Predict joint angles for a single sensor window.

    Args:
        model: trained model
        sensor_sequence: (seq_len, num_sensors) array
        device: torch device
        scaler: fitted StandardScaler (optional)

    Returns:
        (3,) predicted [flexion, abduction, rotation]
    """
    model.eval()
    if scaler is not None:
        sensor_sequence = scaler.transform(sensor_sequence)

    x = torch.FloatTensor(sensor_sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()[0]
