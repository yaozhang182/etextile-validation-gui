"""
Tab 4: Training — Configure hyperparameters, train model, live loss curve.
Uses angle/sensor selections from Tab 3.
"""

import numpy as np
import yaml
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QProgressBar, QMessageBox, QTextEdit,
)
from PyQt6.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.state import ready_subjects

# Label -> value for the alignment dropdown. 'upsample_sensor' interpolates the
# sensor stream onto the per-frame video timestamps, which is what the ground
# truth is defined on.
ALIGNMENT_METHODS = {
    "Align to video frames (recommended)": 'upsample_sensor',
    "Align to sensor samples": 'downsample_labels',
}


class TrainWorker(QThread):
    """Background thread for model training."""
    epoch_done = pyqtSignal(int, int, float)  # epoch, total_epochs, loss
    finished = pyqtSignal(object, list)  # model, losses
    error = pyqtSignal(str)

    def __init__(self, model, train_loader, config, device):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from core.trainer import train_model

            def callback(epoch, epochs, loss):
                self.epoch_done.emit(epoch, epochs, loss)
                if self._stop:
                    raise InterruptedError("Training stopped by user")

            model, losses = train_model(
                self.model, self.train_loader, self.config, self.device,
                progress_callback=callback,
            )
            self.finished.emit(model, losses)
        except InterruptedError:
            self.finished.emit(self.model, [])
        except Exception as e:
            self.error.emit(str(e))


class TrainingPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._worker = None
        self._test_parts = []
        self._losses = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Selection summary ---
        self.selection_label = QLabel("No angles/sensors selected. Go to Tab 3 first.")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        # --- Hyperparameters ---
        hp_group = QGroupBox("Hyperparameters")
        hg = QGridLayout(hp_group)

        hg.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["HybridCNNLSTM", "SimpleCNN1D"])
        hg.addWidget(self.model_combo, 0, 1)

        hg.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        hg.addWidget(self.epochs_spin, 1, 1)

        hg.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        hg.addWidget(self.lr_spin, 2, 1)

        hg.addWidget(QLabel("Sequence Length:"), 3, 0)
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(5, 200)
        self.seq_spin.setValue(40)
        hg.addWidget(self.seq_spin, 3, 1)

        hg.addWidget(QLabel("Batch Size:"), 4, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 256)
        self.batch_spin.setValue(32)
        hg.addWidget(self.batch_spin, 4, 1)

        hg.addWidget(QLabel("Time Alignment:"), 5, 0)
        self.align_combo = QComboBox()
        self.align_combo.addItems(list(ALIGNMENT_METHODS.keys()))
        hg.addWidget(self.align_combo, 5, 1)

        layout.addWidget(hp_group)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # Loss curve
        self.fig = Figure(figsize=(8, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

    def refresh(self):
        sel_a = self.state.get('selected_angles', [])
        sel_s = self.state.get('selected_sensors', [])
        n_train = len(ready_subjects(self.state, 'train'))
        n_test = len(ready_subjects(self.state, 'test'))
        if sel_a and sel_s:
            self.selection_label.setText(
                f"Subjects: {n_train} train, {n_test} test\n"
                f"Targets: {len(sel_a)} angle(s) — {', '.join(sel_a)}\n"
                f"Inputs: {len(sel_s)} sensor(s) — {', '.join(sel_s)}"
            )
        else:
            self.selection_label.setText(
                f"Subjects: {n_train} train, {n_test} test\n"
                "No angles/sensors selected. Go to Tab 3 to select channels."
            )

    def _build_config(self):
        model_map = {"HybridCNNLSTM": "hybrid_cnn_lstm", "SimpleCNN1D": "simple_cnn"}
        return {
            'model_type': model_map[self.model_combo.currentText()],
            'target_cols': self.state.get('selected_angles', ['flexion', 'abduction', 'rotation']),
            'sequence_length': self.seq_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'epochs': self.epochs_spin.value(),
            'alignment_method': ALIGNMENT_METHODS[self.align_combo.currentText()],
            'cnn_filters': 32,
            'lstm_hidden': 64,
            'dropout': 0.1,
        }

    def _align_split(self, split, sel_angles, sel_sensors, method):
        """
        Time-align every ready subject in a split.

        Returns a list of (subject_id, X, Y, t). Subjects that cannot be aligned
        are reported in the log and skipped rather than failing the whole run.
        """
        from core.data_alignment import build_supervised_arrays

        out = []
        for subject in ready_subjects(self.state, split):
            try:
                X, Y, t = build_supervised_arrays(
                    subject['sensor_df'], subject['video_df'], subject['angles'],
                    sel_angles, sel_sensors, method=method,
                )
            except Exception as e:
                self.log.append(f"  [{split}/{subject['id']}] skipped — {e}")
                continue

            if len(X) == 0:
                self.log.append(f"  [{split}/{subject['id']}] skipped — no valid samples")
                continue

            span = (t[-1] - t[0]) if len(t) > 1 else 0.0
            self.log.append(
                f"  [{split}/{subject['id']}] {len(X)} aligned samples "
                f"over {span:.1f} s"
            )
            out.append((subject['id'], X, Y, t))
        return out

    def _start_training(self):
        sel_angles = self.state.get('selected_angles', [])
        sel_sensors = self.state.get('selected_sensors', [])

        if not sel_angles or not sel_sensors:
            QMessageBox.warning(self, "Missing Selection",
                                "Select angle targets and sensor inputs in Tab 3 first.")
            return

        if not ready_subjects(self.state, 'train'):
            QMessageBox.warning(self, "Missing Data",
                                "Add at least one training subject and extract its "
                                "skeleton in Tab 1 first.")
            return

        self.log.clear()

        # Drop results from any previous run, so a failed or test-less run can
        # never leave the dashboard showing stale numbers as if they were new.
        for key in ('model', 'train_losses', 'predictions', 'ground_truth',
                    'metrics', 'metrics_per_subject'):
            self.state[key] = None
        self._test_parts = []

        try:
            config = self._build_config()
            num_targets = len(sel_angles)
            num_sensors = len(sel_sensors)
            config['target_cols'] = sel_angles
            method = config['alignment_method']
            seq_len = config['sequence_length']

            from core.data_alignment import ShoulderDataset
            from sklearn.preprocessing import StandardScaler
            import torch
            from torch.utils.data import DataLoader, ConcatDataset

            # --- Time-align each subject on the shared global clock ---
            self.log.append(f"Aligning data ({method})...")
            train_parts = self._align_split('train', sel_angles, sel_sensors, method)
            test_parts = self._align_split('test', sel_angles, sel_sensors, method)

            if not train_parts:
                QMessageBox.warning(self, "No Usable Data",
                                    "No training subject could be aligned. Check that the "
                                    "sensor and video timestamps overlap (see the log).")
                return

            # Fit the scaler once over all training subjects pooled, then apply
            # the same transform to test subjects.
            scaler = StandardScaler()
            scaler.fit(np.vstack([X for _, X, _, _ in train_parts]))

            # Window each subject separately, then concatenate. Windowing pooled
            # arrays would let a sequence straddle two subjects.
            train_sets, skipped = [], []
            for sid, X, Y, _ in train_parts:
                ds = ShoulderDataset(scaler.transform(X), Y, seq_len)
                if len(ds) < 1:
                    skipped.append(f"{sid} ({len(X)} samples < seq_len {seq_len})")
                    continue
                train_sets.append(ds)

            for note in skipped:
                self.log.append(f"  skipped {note}")

            if not train_sets:
                QMessageBox.warning(
                    self, "Not Enough Data",
                    f"No training subject has more than {seq_len} aligned samples. "
                    f"Reduce the sequence length or record longer sessions."
                )
                return

            train_ds = ConcatDataset(train_sets)
            train_loader = DataLoader(
                train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0
            )

            self.log.append(
                f"Train windows: {len(train_ds)} from {len(train_sets)} subject(s), "
                f"Sensors: {num_sensors}, Targets: {num_targets}"
            )

            # Create model
            from core.model import create_model
            config['target_cols'] = sel_angles  # for len() in create_model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = create_model(config, num_sensors=num_sensors).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            self.log.append(f"Model: {model.__class__.__name__}, {total_params:,} params, device={device}")

            # Store for results tab
            self.state['config'] = config
            self.state['scaler'] = scaler
            self.state['alignment_method'] = method
            self._test_parts = [
                (sid, scaler.transform(X), Y) for sid, X, Y, _ in test_parts
            ]
            self._losses = []

            # Start training thread
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(config['epochs'])

            self._worker = TrainWorker(model, train_loader, config, device)
            self._worker.epoch_done.connect(self._on_epoch)
            self._worker.finished.connect(self._on_training_done)
            self._worker.error.connect(self._on_training_error)
            self._worker.start()

        except Exception as e:
            self.log.append(f"ERROR: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def _stop_training(self):
        if self._worker:
            self._worker.stop()

    def _on_epoch(self, epoch, total, loss):
        self._losses.append(loss)
        self.progress_bar.setValue(epoch)
        self.progress_label.setText(f"Epoch {epoch}/{total} — Loss: {loss:.4f}")

        # Update loss curve
        self.ax.clear()
        self.ax.plot(range(1, len(self._losses) + 1), self._losses, 'b-', linewidth=1.5)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss (MSE)")
        self.ax.set_title("Training Loss")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_training_done(self, model, losses):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.state['model'] = model
        self.state['train_losses'] = self._losses

        # Run evaluation if test data is available
        if self._test_parts:
            self._run_evaluation(model)
        else:
            self.log.append("Training complete. No test subjects for evaluation.")
            self.progress_label.setText("Training complete. No test data.")

    def _run_evaluation(self, model):
        """Evaluate each test subject separately, then pool for global metrics."""
        try:
            import torch
            from torch.utils.data import DataLoader
            from core.data_alignment import ShoulderDataset
            from core.trainer import evaluate_model
            from core.evaluator import calculate_metrics

            config = self.state['config']
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            seq_len = config['sequence_length']
            sel_angles = self.state.get('selected_angles', [])
            joint_names = [n.replace('_', ' ').title() for n in sel_angles]

            per_subject, all_preds, all_gt = {}, [], []

            for sid, X, Y in self._test_parts:
                test_ds = ShoulderDataset(X, Y, seq_len)
                if len(test_ds) < 1:
                    self.log.append(
                        f"  [test/{sid}] skipped — {len(X)} samples < seq_len {seq_len}"
                    )
                    continue

                test_loader = DataLoader(test_ds, batch_size=config['batch_size'] * 2,
                                         shuffle=False, num_workers=0)
                preds, gt = evaluate_model(model, test_loader, device)
                per_subject[sid] = calculate_metrics(gt, preds, joint_names)
                all_preds.append(preds)
                all_gt.append(gt)

                self.log.append(
                    f"  [test/{sid}] MPJAE={per_subject[sid]['Global_MPJAE']:.2f} deg, "
                    f"PCC={per_subject[sid]['Global_PCC']:.3f}"
                )

            if not all_preds:
                self.log.append("No test subject had enough samples for evaluation.")
                self.progress_label.setText("Training complete. Test sets too small.")
                return

            preds = np.concatenate(all_preds, axis=0)
            gt = np.concatenate(all_gt, axis=0)
            metrics = calculate_metrics(gt, preds, joint_names)

            self.state['predictions'] = preds
            self.state['ground_truth'] = gt
            self.state['metrics'] = metrics
            self.state['metrics_per_subject'] = per_subject

            self.log.append(
                f"Pooled over {len(per_subject)} subject(s): "
                f"MPJAE={metrics['Global_MPJAE']:.2f} deg, PCC={metrics['Global_PCC']:.3f}"
            )
            self.progress_label.setText(
                f"Done! MPJAE={metrics['Global_MPJAE']:.2f} deg — Switch to Tab 5 for details."
            )

        except Exception as e:
            self.log.append(f"Evaluation error: {e}")

    def _on_training_error(self, msg):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Training Error", msg)
