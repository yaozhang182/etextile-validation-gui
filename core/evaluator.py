"""
Evaluation and visualization module.

Metrics: MPJAE, AMPE, RMSE, PCC
Visualizations: error heatmap, sensor importance, prediction curves
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path


def calculate_metrics(y_true, y_pred, joint_names=None):
    """
    Calculate comprehensive metrics for joint angle prediction.

    Args:
        y_true: (N, 3) ground truth angles
        y_pred: (N, 3) predicted angles
        joint_names: list of names (default: ['Flexion', 'Abduction', 'Rotation'])

    Returns:
        dict with global and per-joint metrics
    """
    if joint_names is None:
        joint_names = ['Flexion', 'Abduction', 'Rotation']

    n_joints = y_true.shape[1]
    metrics = {}

    # MPJAE
    abs_error = np.abs(y_true - y_pred)
    mpjae_per_joint = np.mean(abs_error, axis=0)
    metrics['Global_MPJAE'] = float(np.mean(mpjae_per_joint))
    metrics['MPJAE_per_joint'] = {
        joint_names[i]: float(mpjae_per_joint[i]) for i in range(n_joints)
    }

    # AMPE (normalized by range of motion, min 5 deg)
    ranges = np.maximum(np.max(y_true, axis=0) - np.min(y_true, axis=0), 5.0)
    ampe_per_joint = (mpjae_per_joint / ranges) * 100
    metrics['Global_AMPE'] = float(np.mean(ampe_per_joint))
    metrics['AMPE_per_joint'] = {
        joint_names[i]: float(ampe_per_joint[i]) for i in range(n_joints)
    }

    # RMSE
    squared_error = (y_true - y_pred) ** 2
    rmse_per_joint = np.sqrt(np.mean(squared_error, axis=0))
    metrics['Global_RMSE'] = float(np.sqrt(np.mean(squared_error)))
    metrics['RMSE_per_joint'] = {
        joint_names[i]: float(rmse_per_joint[i]) for i in range(n_joints)
    }

    # PCC
    pccs = []
    for i in range(n_joints):
        if np.std(y_pred[:, i]) < 1e-5:
            pccs.append(0.0)
        else:
            try:
                corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
                pccs.append(corr if not np.isnan(corr) else 0.0)
            except Exception:
                pccs.append(0.0)

    metrics['Global_PCC'] = float(np.mean(pccs))
    metrics['PCC_per_joint'] = {
        joint_names[i]: float(pccs[i]) for i in range(n_joints)
    }

    metrics['MAE_std'] = float(np.std(abs_error))
    metrics['Max_error'] = float(np.max(abs_error))
    metrics['Range_of_motion'] = {
        joint_names[i]: float(ranges[i]) for i in range(n_joints)
    }

    return metrics


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_error_heatmap(mpjae_per_joint, joint_names, save_path=None):
    """
    Generate error heatmap figure.

    Returns:
        matplotlib Figure (also saves to save_path if provided)
    """
    error_matrix = np.asarray(mpjae_per_joint).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        error_matrix, annot=True, fmt=".2f", cmap="Reds",
        xticklabels=joint_names, yticklabels=['Test Motion'],
        cbar_kws={'label': 'MPJAE (degrees)'}, vmin=0, ax=ax,
    )
    ax.set_title("Error Attribution Heatmap", fontsize=14, fontweight='bold')
    ax.set_xlabel("Clinical Joint Angles", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sensor_importance(model, sensor_names, save_path=None):
    """
    Generate sensor importance bar chart.

    Returns:
        matplotlib Figure
    """
    importance = model.get_feature_importance()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(importance / importance.max())
    bars = ax.bar(sensor_names, importance, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, importance):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title("Sensor Importance Score", fontsize=14, fontweight='bold')
    ax.set_ylabel("Normalized Weight Contribution", fontsize=12)
    ax.set_xlabel("Sensor Position", fontsize=12)
    ax.set_ylim(0, max(importance) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_curves(y_true, y_pred, joint_names, save_path=None, timestamps=None):
    """
    Plot prediction vs ground truth curves for all joints.

    Returns:
        matplotlib Figure
    """
    n_joints = y_true.shape[1]
    if timestamps is None:
        timestamps = np.arange(len(y_true))
    if timestamps[0] > 1000:
        timestamps = timestamps - timestamps[0]

    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 10))
    if n_joints == 1:
        axes = [axes]

    for i in range(n_joints):
        ax = axes[i]
        ax.plot(timestamps, y_true[:, i], label='Ground Truth',
                color='black', linewidth=2, alpha=0.7)
        ax.plot(timestamps, y_pred[:, i], label='Prediction',
                color='red', linewidth=1.5, linestyle='--', alpha=0.8)

        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        corr = 0.0
        if np.std(y_pred[:, i]) > 1e-5:
            try:
                corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            except Exception:
                pass

        ax.set_ylabel(f'{joint_names[i]} (deg)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{joint_names[i]} - MAE: {mae:.2f} | Corr: {corr:.3f}', fontsize=10)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle('Joint Angle Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Reports
# =============================================================================

def generate_report(metrics, output_dir, model=None, sensor_names=None):
    """Generate text metrics report."""
    report_path = Path(output_dir) / 'metrics_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("E-TEXTILE VALIDATION - METRICS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("GLOBAL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"MPJAE: {metrics['Global_MPJAE']:.3f} deg\n")
        f.write(f"AMPE: {metrics['Global_AMPE']:.2f}%\n")
        f.write(f"RMSE: {metrics['Global_RMSE']:.3f} deg\n")
        f.write(f"PCC: {metrics['Global_PCC']:.4f}\n\n")

        f.write("PER-JOINT METRICS\n")
        f.write("-" * 70 + "\n\n")

        for joint in metrics['MPJAE_per_joint']:
            f.write(f"{joint.upper()}:\n")
            f.write(f"  MPJAE: {metrics['MPJAE_per_joint'][joint]:.3f} deg\n")
            f.write(f"  AMPE: {metrics['AMPE_per_joint'][joint]:.2f}%\n")
            f.write(f"  RMSE: {metrics['RMSE_per_joint'][joint]:.3f} deg\n")
            f.write(f"  PCC: {metrics['PCC_per_joint'][joint]:.4f}\n")
            f.write(f"  ROM: {metrics['Range_of_motion'][joint]:.2f} deg\n\n")

        if model is not None and sensor_names is not None:
            f.write("SENSOR IMPORTANCE\n")
            f.write("-" * 70 + "\n\n")
            importance = model.get_feature_importance()
            ranked = sorted(zip(sensor_names, importance), key=lambda x: x[1], reverse=True)
            for rank, (sensor, imp) in enumerate(ranked, 1):
                f.write(f"  #{rank} {sensor}: {imp:.4f} ({imp * 100:.2f}%)\n")

        f.write("\n" + "=" * 70 + "\n")

    return report_path


def save_metrics_csv(metrics, output_dir):
    """Save metrics to CSV."""
    csv_data = {
        'Global_MPJAE': metrics['Global_MPJAE'],
        'Global_AMPE': metrics['Global_AMPE'],
        'Global_RMSE': metrics['Global_RMSE'],
        'Global_PCC': metrics['Global_PCC'],
    }
    for joint in metrics['MPJAE_per_joint']:
        csv_data[f'{joint}_MPJAE'] = metrics['MPJAE_per_joint'][joint]
        csv_data[f'{joint}_AMPE'] = metrics['AMPE_per_joint'][joint]
        csv_data[f'{joint}_RMSE'] = metrics['RMSE_per_joint'][joint]
        csv_data[f'{joint}_PCC'] = metrics['PCC_per_joint'][joint]

    df = pd.DataFrame([csv_data])
    csv_path = Path(output_dir) / 'metrics_summary.csv'
    df.to_csv(csv_path, index=False)
    return csv_path
