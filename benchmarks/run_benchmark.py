#!/usr/bin/env python3
"""
End-to-end timing benchmark for the toolchain.

Produces the processing-time table the reviewers asked for: how long each stage
takes on ordinary hardware, so the "feedback within minutes" claim in the paper
is backed by a measurement rather than an estimate.

Stages timed separately:

  1. pose extraction     MediaPipe over every video frame (dominant cost)
  2. joint angles        inverse kinematics over the landmark sequence
  3. time alignment      interpolation onto the shared clock
  4. training            N epochs of the selected model
  5. evaluation          inference over the test set

Usage:

    python benchmarks/run_benchmark.py --data-dir input_data/examples_data
    python benchmarks/run_benchmark.py --data-dir <dir> --model-complexity 1 --epochs 20
    python benchmarks/run_benchmark.py --data-dir <dir> --extract-frames 300   # quick

The directory must contain session_N/ subdirectories, each holding video.mp4,
sensor.csv and frames.csv -- the layout of input_data/examples_data. Choose which
sessions train and which are held out with --train-sessions / --test-sessions.

Results are printed as a table and written to benchmarks/results_<host>.json.
Report the machine spec alongside the numbers — they are hardware-dependent.
"""

import argparse
import json
import platform
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET_ANGLES = ['right_shoulder_flexion', 'right_shoulder_abduction',
                 'right_shoulder_rotation']


class Timings(dict):
    """Measured values. `stages` holds durations; `info` holds everything else."""

    def __init__(self):
        super().__init__()
        self.stages = []      # ordered stage names, so the report keeps pipeline order

    @contextmanager
    def stage(self, name, verbose=True):
        if verbose:
            print(f"  {name}...", flush=True)
        t0 = time.perf_counter()
        yield
        self[name] = time.perf_counter() - t0
        self.stages.append(name)
        if verbose:
            print(f"    {self[name]:.2f} s", flush=True)


def describe_machine():
    try:
        import torch
        torch_threads = torch.get_num_threads()
    except Exception:
        torch_threads = None

    cpu_model = platform.processor()
    try:  # /proc/cpuinfo gives a far more useful name on Linux
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu_model = line.split(':', 1)[1].strip()
                break
    except Exception:
        pass

    import os
    return {
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'cpu': cpu_model,
        'cpu_count': os.cpu_count(),
        'torch_threads': torch_threads,
    }


def extract(video_path, timings, complexity, max_frames, label):
    """Time MediaPipe extraction and the joint-angle computation separately."""
    import cv2
    import mediapipe as mp_lib
    from core.smpl_extraction import mediapipe_to_smplh22
    from core.joint_angles import extract_all_joint_angles

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp_lib.solutions.pose.Pose(
        static_image_mode=False, model_complexity=complexity,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    landmarks, visibility, n = [], [], 0
    with timings.stage(f'{label}: pose extraction'):
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                lm = res.pose_world_landmarks.landmark
                landmarks.append(np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32))
                visibility.append(np.array([l.visibility for l in lm], dtype=np.float32))
            else:
                landmarks.append(np.full((33, 3), np.nan, dtype=np.float32))
                visibility.append(np.zeros(33, dtype=np.float32))
            n += 1
    cap.release()
    pose.close()

    if not landmarks:
        raise RuntimeError(f"no frames read from {video_path}")

    arr = np.stack(landmarks)
    with timings.stage(f'{label}: joint angles'):
        smplh = mediapipe_to_smplh22(arr)
        angles, _ = extract_all_joint_angles(smplh, fps=fps)

    timings[f'{label}: frames processed'] = n
    timings[f'{label}: video frames total'] = total
    timings[f'{label}: fps'] = fps
    return angles, np.stack(visibility), fps, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True,
                    help='dataset root containing session_* directories')
    ap.add_argument('--train-sessions', default='1,2,4',
                    help='comma-separated session numbers used for training')
    ap.add_argument('--test-sessions', default='3',
                    help='comma-separated session numbers held out for testing')
    ap.add_argument('--model-complexity', type=int, default=2, choices=[0, 1, 2],
                    help='MediaPipe Pose complexity: 0 lite, 1 full, 2 heavy (GUI default)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seq-len', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--extract-frames', type=int, default=None,
                    help='cap frames per video (for a quick run; omit for the real number)')
    ap.add_argument('--alignment', default='upsample_sensor',
                    choices=['upsample_sensor', 'downsample_labels'])
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    machine = describe_machine()
    print(f"Machine: {machine['cpu']} ({machine['cpu_count']} cores)")
    print(f"MediaPipe model_complexity={args.model_complexity}, "
          f"{args.epochs} epochs, seq_len={args.seq_len}\n")

    timings = Timings()

    # --- 1 & 2: extraction + inverse kinematics -------------------------
    from core.data_alignment import estimate_fps_from_frames

    splits = {
        'train': [int(x) for x in args.train_sessions.split(',') if x.strip()],
        'test': [int(x) for x in args.test_sessions.split(',') if x.strip()],
    }
    print("Sessions: " + ", ".join(f"{k}={v}" for k, v in splits.items()) + "\n")

    data = {'train': [], 'test': []}
    for split, session_ids in splits.items():
        for sid in session_ids:
            session_dir = data_dir / f'session_{sid}'
            if not session_dir.is_dir():
                raise FileNotFoundError(f"no such session directory: {session_dir}")

            label = f'{split}/session_{sid}'
            angles, visibility, fps, n = extract(
                session_dir / 'video.mp4', timings,
                args.model_complexity, args.extract_frames, label,
            )
            video_df = pd.read_csv(session_dir / 'frames.csv', sep=';')
            data[split].append({
                'id': f'session_{sid}',
                'angles': angles,
                'sensor_df': pd.read_csv(session_dir / 'sensor.csv', sep=';'),
                'video_df': video_df,
                'visibility': visibility,
                # The container's fps tag is unreliable; timestamps are authoritative.
                'fps': estimate_fps_from_frames(video_df, fallback=fps),
            })

    # --- 3: time alignment ----------------------------------------------
    from core.data_alignment import build_supervised_arrays, ShoulderDataset

    exclude = {'epochtime', 'epoctime', 'time', 'timestamp', 'frameindex', 'marker'}
    sensor_cols = [c for c in data['train'][0]['sensor_df'].columns
                   if c.lower() not in exclude]
    # A channel missing from any session would break the pooled model.
    for split, sessions in data.items():
        for d in sessions:
            missing = [c for c in sensor_cols if c not in d['sensor_df'].columns]
            if missing:
                raise ValueError(f"{d['id']} is missing sensor channels {missing}")

    with timings.stage('time alignment'):
        aligned = {
            split: [
                (d['id'], *build_supervised_arrays(
                    d['sensor_df'], d['video_df'], d['angles'],
                    TARGET_ANGLES, sensor_cols, method=args.alignment,
                ))
                for d in sessions
            ]
            for split, sessions in data.items()
        }

    for split, parts in aligned.items():
        total_samples = sum(len(X) for _, X, _, _ in parts)
        timings[f'{split}: aligned samples'] = total_samples
        for sid, X, _, t in parts:
            span = (t[-1] - t[0]) if len(t) > 1 else 0.0
            timings[f'{split}/{sid}: aligned span (s)'] = round(float(span), 1)
            print(f"  {split}/{sid}: {len(X)} samples over {span:.1f} s")

    # --- 4: training ------------------------------------------------------
    import torch
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler
    from core.model import create_model
    from core.trainer import train_model, evaluate_model
    from core.evaluator import calculate_metrics

    # Fit the scaler once over all training sessions pooled, then window each
    # session separately so no sliding window straddles two recordings.
    from torch.utils.data import ConcatDataset

    scaler = StandardScaler().fit(np.vstack([X for _, X, _, _ in aligned['train']]))
    train_sets = [ShoulderDataset(scaler.transform(X), Y, args.seq_len)
                  for _, X, Y, _ in aligned['train']]
    train_sets = [ds for ds in train_sets if len(ds) > 0]
    test_sets = [(sid, ShoulderDataset(scaler.transform(X), Y, args.seq_len))
                 for sid, X, Y, _ in aligned['test']]
    test_sets = [(sid, ds) for sid, ds in test_sets if len(ds) > 0]

    if not train_sets or not test_sets:
        print(f"\nERROR: no session has more than seq_len={args.seq_len} aligned "
              f"samples (train sets {len(train_sets)}, test sets {len(test_sets)}).")
        return 1
    train_ds = ConcatDataset(train_sets)

    config = {
        'model_type': 'hybrid_cnn_lstm', 'target_cols': TARGET_ANGLES,
        'sequence_length': args.seq_len, 'batch_size': args.batch_size,
        'learning_rate': 0.001, 'epochs': args.epochs,
        'cnn_filters': 32, 'lstm_hidden': 64, 'dropout': 0.1,
    }
    model = create_model(config, num_sensors=len(sensor_cols))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    with timings.stage('training'):
        model, _ = train_model(model, loader, config, 'cpu')

    # --- 5: evaluation ----------------------------------------------------
    with timings.stage('evaluation'):
        all_preds, all_gt = [], []
        for sid, ds in test_sets:
            loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False)
            p, g = evaluate_model(model, loader, 'cpu')
            all_preds.append(p)
            all_gt.append(g)
        preds = np.concatenate(all_preds)
        gt = np.concatenate(all_gt)
    metrics = calculate_metrics(gt, preds, TARGET_ANGLES)

    # --- confidence of the ground truth used ------------------------------
    from core.confidence import tracking_report
    conf = {
        f"{split}/{d['id']}": tracking_report({'visibility': d['visibility']}, TARGET_ANGLES)
        for split, sessions in data.items() for d in sessions
    }

    # --- report -----------------------------------------------------------
    extraction_total = sum(timings[k] for k in timings.stages
                           if k.endswith('pose extraction'))
    frames_total = sum(timings[k] for k in timings if k.endswith('frames processed'))

    print("\n" + "=" * 62)
    print(f"{'Stage':<34}{'Seconds':>12}{'Share':>10}")
    print("-" * 62)
    total = sum(timings[k] for k in timings.stages)
    for k in timings.stages:
        print(f"{k:<34}{timings[k]:>12.2f}{timings[k] / total * 100:>9.1f}%")
    print("-" * 62)
    print(f"{'TOTAL':<34}{total:>12.2f}")
    print("=" * 62)

    per_frame = extraction_total / frames_total * 1000 if frames_total else float('nan')
    print(f"\nPose extraction: {per_frame:.0f} ms/frame over {int(frames_total)} frames")
    print(f"Aligned samples: train {timings['train: aligned samples']} "
          f"({len(train_sets)} session(s)), "
          f"test {timings['test: aligned samples']} ({len(test_sets)} session(s))")
    print(f"Accuracy on this data: MPJAE {metrics['Global_MPJAE']:.2f} deg, "
          f"PCC {metrics['Global_PCC']:.3f}")
    for name, rep in conf.items():
        if rep.get('available'):
            print(f"Ground-truth tracking confidence ({name}): "
                  f"{rep['overall']['mean']:.2f} ({rep['label']})")
    print("\nNOTE: accuracy depends entirely on the garment and recording; it is "
          "reported here only so the timing run is not mistaken for a validation.")

    payload = {
        'machine': machine,
        'settings': vars(args),
        'stage_seconds': {k: timings[k] for k in timings.stages},
        'measurements': {k: v for k, v in timings.items() if k not in timings.stages},
        'ms_per_frame_extraction': per_frame,
        'metrics': {k: float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float))},
        'ground_truth_confidence': {
            s: (r['overall'] if r.get('available') else None) for s, r in conf.items()
        },
    }
    out = Path(args.output or
               Path(__file__).parent / f"results_{machine['host']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
