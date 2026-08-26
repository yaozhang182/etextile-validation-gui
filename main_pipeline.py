#!/usr/bin/env python3
"""
E-Textile Validation - CLI Pipeline

End-to-end pipeline for testing the core modules without GUI.

Usage:
    python main_pipeline.py --data-dir /path/to/iteration_folder
    python main_pipeline.py --data-dir ./input_data/Iteration_1 --config config/default.yaml
"""

import argparse
import time
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from core.data_alignment import load_iteration_data, ShoulderDataset, save_metadata
from core.model import create_model
from core.trainer import train_model, evaluate_model
from core.evaluator import (
    calculate_metrics,
    plot_error_heatmap,
    plot_sensor_importance,
    plot_prediction_curves,
    generate_report,
    save_metrics_csv,
)


def main():
    parser = argparse.ArgumentParser(description='E-Textile Validation Pipeline')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to iteration data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data-dir/results)')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    start_time = time.time()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Step 1: Load data
    print("\n[1/5] Loading and aligning data...")
    train_sensor, train_labels, test_sensor, test_labels, scaler = \
        load_iteration_data(data_dir, alignment_method=config.get('alignment_method', 'downsample_labels'))

    seq_len = config['sequence_length']
    train_dataset = ShoulderDataset(train_sensor, train_labels, seq_len)
    test_dataset = ShoulderDataset(test_sensor, test_labels, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'] * 2,
                             shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")
    print(f"  Sensors: {train_sensor.shape[1]} | Seq len: {seq_len}")

    # Step 2: Create model
    print("\n[2/5] Setting up model...")
    num_sensors = train_sensor.shape[1]
    model = create_model(config, num_sensors).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {model.__class__.__name__} | {total_params:,} parameters")

    # Step 3: Train
    print("\n[3/5] Training...")
    def print_progress(epoch, epochs, loss):
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {loss:.6f}")

    model, train_losses = train_model(
        model, train_loader, config, device, output_dir,
        progress_callback=print_progress
    )

    # Step 4: Evaluate
    print("\n[4/5] Evaluating...")
    predictions, ground_truth = evaluate_model(model, test_loader, device)

    joint_names = [n.capitalize() for n in config['target_cols']]
    metrics = calculate_metrics(ground_truth, predictions, joint_names)

    print(f"  MPJAE: {metrics['Global_MPJAE']:.3f} deg")
    print(f"  AMPE:  {metrics['Global_AMPE']:.2f}%")
    print(f"  RMSE:  {metrics['Global_RMSE']:.3f} deg")
    print(f"  PCC:   {metrics['Global_PCC']:.4f}")

    # Step 5: Save results
    print("\n[5/5] Saving results...")
    sensor_names = [f'S{i+1}' for i in range(num_sensors)]

    mpjae_array = np.array([metrics['MPJAE_per_joint'][j] for j in joint_names])
    plot_error_heatmap(mpjae_array, joint_names, output_dir / 'error_heatmap.png')
    plot_sensor_importance(model, sensor_names, output_dir / 'sensor_importance.png')
    plot_prediction_curves(ground_truth, predictions, joint_names,
                           output_dir / 'prediction_curves.png')

    save_metrics_csv(metrics, output_dir)
    generate_report(metrics, output_dir, model=model, sensor_names=sensor_names)
    np.save(output_dir / 'predictions.npy', predictions)
    np.save(output_dir / 'ground_truth.npy', ground_truth)

    elapsed = time.time() - start_time
    save_metadata(output_dir, {
        'num_sensors': int(num_sensors),
        'train_samples': int(len(train_dataset)),
        'test_samples': int(len(test_dataset)),
        'processing_time_seconds': float(elapsed),
        'device': device,
        'metrics': {
            'MPJAE': float(metrics['Global_MPJAE']),
            'AMPE': float(metrics['Global_AMPE']),
            'RMSE': float(metrics['Global_RMSE']),
            'PCC': float(metrics['Global_PCC']),
        },
    })

    print(f"\nDone in {elapsed:.1f}s. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
