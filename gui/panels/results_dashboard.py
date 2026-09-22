"""
Tab 5: Results Dashboard — Metrics tables, prediction curves, sensor
importance, export.

The per-joint error heatmap that used to sit here was dropped: it was a single
row of numbers already printed in the metrics table right above it, so it spent
half the width restating what the reader had just read. The prediction-vs-ground-
truth curves take that space instead — they are the figure people actually read
to judge whether the garment tracks the movement — and sensor importance sits
below them at the size a ten-bar chart needs.
"""

import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QFileDialog, QGroupBox, QMessageBox,
    QScrollArea,
)
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui import help as help_ui, icons


class ResultsDashboardPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        headline_row = QHBoxLayout()
        self.info_label = QLabel("Train a model in Tab 4 first.")
        self.info_label.setProperty("headline", True)
        headline_row.addWidget(self.info_label)
        help_ui.attach(headline_row, 'headline_metrics')
        headline_row.addStretch()
        layout.addLayout(headline_row)

        # --- Per-subject breakdown ---
        self.subject_group = QGroupBox("Per-Subject Results (test)")
        sg = QVBoxLayout(self.subject_group)
        sg.addWidget(help_ui.labelled(
            "Each test session scored on its own", 'per_subject'))
        self.subject_table = QTableWidget()
        self.subject_table.setMaximumHeight(150)
        sg.addWidget(self.subject_table)
        layout.addWidget(self.subject_group)

        # --- Metrics table ---
        metrics_group = QGroupBox("Metrics Summary (pooled over test subjects)")
        mg = QVBoxLayout(metrics_group)
        mg.addWidget(help_ui.labelled(
            "The last column qualifies all the others",
            'tracking_conf_column'))
        self.metrics_table = QTableWidget()
        self.metrics_table.setMinimumHeight(150)
        self.metrics_table.setMaximumHeight(240)
        mg.addWidget(self.metrics_table)
        layout.addWidget(metrics_group)

        # --- Prediction curves: the headline figure, full width ---
        pred_group = QGroupBox("Prediction vs Ground Truth")
        pg = QVBoxLayout(pred_group)
        pg.addWidget(help_ui.labelled(
            "One panel per predicted channel, over the test recording",
            'prediction_curves'))
        self.fig_pred = Figure(figsize=(11, 5.6))
        self.canvas_pred = FigureCanvas(self.fig_pred)
        # One row of small-multiples needs roughly this much to stay readable;
        # more channels add rows and the scroll area takes up the slack.
        self.canvas_pred.setMinimumHeight(430)
        pg.addWidget(self.canvas_pred)
        layout.addWidget(pred_group)

        # --- Sensor importance: a ten-bar chart, so kept modest ---
        importance_group = QGroupBox("Sensor Importance")
        ig = QVBoxLayout(importance_group)
        ig.addWidget(help_ui.labelled(
            "What sensor importance tells you about the layout",
            'sensor_importance'))
        importance_row = QHBoxLayout()
        self.fig_importance = Figure(figsize=(5.5, 2.4))
        self.ax_importance = self.fig_importance.add_subplot(111)
        self.canvas_importance = FigureCanvas(self.fig_importance)
        self.canvas_importance.setMinimumHeight(200)
        self.canvas_importance.setMaximumHeight(240)
        self.canvas_importance.setMaximumWidth(640)
        importance_row.addWidget(self.canvas_importance)
        importance_row.addStretch()
        ig.addLayout(importance_row)
        layout.addWidget(importance_group)

        # --- Export buttons ---
        export_layout = QHBoxLayout()
        btn_csv = icons.decorate(QPushButton("Export Metrics CSV"), 'download')
        btn_csv.clicked.connect(self._export_csv)
        export_layout.addWidget(btn_csv)

        btn_report = icons.decorate(QPushButton("Export Report"), 'download')
        btn_report.clicked.connect(self._export_report)
        export_layout.addWidget(btn_report)

        btn_plots = icons.decorate(QPushButton("Save All Plots"), 'download')
        btn_plots.clicked.connect(self._save_plots)
        export_layout.addWidget(btn_plots)
        layout.addLayout(export_layout)

        layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

    def refresh(self):
        metrics = self.state.get('metrics')
        if metrics is None:
            self.info_label.setText("No results yet. Train a model in Tab 4.")
            return

        self.info_label.setText(
            f"MPJAE: {metrics['Global_MPJAE']:.2f} deg  |  "
            f"AMPE: {metrics['Global_AMPE']:.1f}%  |  "
            f"RMSE: {metrics['Global_RMSE']:.2f} deg  |  "
            f"PCC: {metrics['Global_PCC']:.3f}"
        )

        self._fill_subject_table()
        self._fill_metrics_table(metrics)
        self._plot_predictions(metrics)
        self._plot_sensor_importance()

    def reset(self):
        """Empty every table and figure so no stale result is left on screen."""
        self.info_label.setText("Train a model in Tab 4 first.")
        for table in (self.subject_table, self.metrics_table):
            table.clearContents()
            table.setRowCount(0)
        self.subject_group.setVisible(False)
        for figure, canvas in ((self.fig_importance, self.canvas_importance),
                               (self.fig_pred, self.canvas_pred)):
            figure.clf()
            canvas.draw()
        # clf() dropped the axes this attribute referred to
        self.ax_importance = self.fig_importance.add_subplot(111)

    def _fill_subject_table(self):
        """Show how the model does on each test subject individually."""
        per_subject = self.state.get('metrics_per_subject') or {}

        # With a single test subject this duplicates the pooled table.
        self.subject_group.setVisible(len(per_subject) > 1)
        if len(per_subject) <= 1:
            return

        cols = ['MPJAE (deg)', 'AMPE (%)', 'RMSE (deg)', 'PCC']
        subject_ids = list(per_subject.keys())
        self.subject_table.setRowCount(len(subject_ids))
        self.subject_table.setColumnCount(len(cols))
        self.subject_table.setHorizontalHeaderLabels(cols)
        self.subject_table.setVerticalHeaderLabels(subject_ids)

        for row, sid in enumerate(subject_ids):
            m = per_subject[sid]
            for col, (key, fmt) in enumerate([
                ('Global_MPJAE', '{:.2f}'), ('Global_AMPE', '{:.1f}'),
                ('Global_RMSE', '{:.2f}'), ('Global_PCC', '{:.3f}'),
            ]):
                self.subject_table.setItem(row, col, QTableWidgetItem(fmt.format(m[key])))

        self.subject_table.resizeColumnsToContents()

    def _target_confidence(self):
        """
        Mean tracking confidence per selected target channel, averaged over the
        test subjects the metrics were computed from.
        """
        from core.confidence import tracking_report
        from gui.state import ready_subjects

        sel = self.state.get('selected_angles', [])
        if not sel:
            return {}

        collected = {}
        for subject in ready_subjects(self.state, 'test'):
            report = tracking_report(subject.get('skeleton'), sel)
            if not report.get('available'):
                continue
            for name, summary in report['per_angle'].items():
                collected.setdefault(name, []).append(summary['mean'])

        return {n: float(np.mean(v)) for n, v in collected.items() if v}

    def _fill_metrics_table(self, metrics):
        from core.confidence import LOW_CONFIDENCE

        joint_names = list(metrics['MPJAE_per_joint'].keys())
        # Metric tables are keyed by display names ("Right Shoulder Flexion"),
        # confidence by channel names ("right_shoulder_flexion").
        sel = self.state.get('selected_angles', [])
        display_to_channel = {n.replace('_', ' ').title(): n for n in sel}
        confidence = self._target_confidence()

        cols = ['MPJAE (deg)', 'AMPE (%)', 'RMSE (deg)', 'PCC', 'Tracking conf.']
        self.metrics_table.setRowCount(len(joint_names) + 1)
        self.metrics_table.setColumnCount(len(cols))
        self.metrics_table.setHorizontalHeaderLabels(cols)
        self.metrics_table.setVerticalHeaderLabels(['Global'] + joint_names)

        # Global row
        self.metrics_table.setItem(0, 0, QTableWidgetItem(f"{metrics['Global_MPJAE']:.2f}"))
        self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{metrics['Global_AMPE']:.1f}"))
        self.metrics_table.setItem(0, 2, QTableWidgetItem(f"{metrics['Global_RMSE']:.2f}"))
        self.metrics_table.setItem(0, 3, QTableWidgetItem(f"{metrics['Global_PCC']:.3f}"))
        if confidence:
            self.metrics_table.setItem(
                0, 4, QTableWidgetItem(f"{np.mean(list(confidence.values())):.2f}")
            )

        for i, j in enumerate(joint_names):
            self.metrics_table.setItem(i + 1, 0, QTableWidgetItem(f"{metrics['MPJAE_per_joint'][j]:.2f}"))
            self.metrics_table.setItem(i + 1, 1, QTableWidgetItem(f"{metrics['AMPE_per_joint'][j]:.1f}"))
            self.metrics_table.setItem(i + 1, 2, QTableWidgetItem(f"{metrics['RMSE_per_joint'][j]:.2f}"))
            self.metrics_table.setItem(i + 1, 3, QTableWidgetItem(f"{metrics['PCC_per_joint'][j]:.3f}"))

            conf = confidence.get(display_to_channel.get(j, ''))
            if conf is not None:
                item = QTableWidgetItem(f"{conf:.2f}")
                if conf < LOW_CONFIDENCE:
                    item.setForeground(QColor('#c62828'))
                    item.setToolTip(
                        "The ground truth for this channel is poorly tracked, so "
                        "its error is not a reliable measure of sensor quality."
                    )
                self.metrics_table.setItem(i + 1, 4, item)

        self.metrics_table.resizeColumnsToContents()

    def _plot_sensor_importance(self):
        self.ax_importance.clear()
        model = self.state.get('model')
        sel_sensors = self.state.get('selected_sensors', [])

        if model is None or not hasattr(model, 'get_feature_importance'):
            self.ax_importance.text(0.5, 0.5, "No model", transform=self.ax_importance.transAxes,
                                    ha='center', va='center')
            self.canvas_importance.draw()
            return

        importance = model.get_feature_importance()
        names = sel_sensors if len(sel_sensors) == len(importance) else [f'S{i+1}' for i in range(len(importance))]

        colors = ['#2196F3' if v >= np.mean(importance) else '#90CAF9' for v in importance]
        bars = self.ax_importance.bar(names, importance, color=colors, edgecolor='black')
        for bar, val in zip(bars, importance):
            self.ax_importance.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        self.ax_importance.set_ylabel("Weight")
        self.ax_importance.grid(axis='y', alpha=0.3)
        self.fig_importance.tight_layout()
        self.canvas_importance.draw()

    def _plot_predictions(self, metrics):
        self.fig_pred.clear()

        preds = self.state.get('predictions')
        gt = self.state.get('ground_truth')
        if preds is None or gt is None:
            return

        joint_names = list(metrics['MPJAE_per_joint'].keys())
        n_joints = len(joint_names)
        n_cols = min(n_joints, 3)
        n_rows = (n_joints + n_cols - 1) // n_cols

        for i, jname in enumerate(joint_names):
            ax = self.fig_pred.add_subplot(n_rows, n_cols, i + 1)
            ax.plot(gt[:, i], 'k-', linewidth=1.2, label='Ground Truth', alpha=0.8)
            ax.plot(preds[:, i], 'r--', linewidth=1, label='Prediction', alpha=0.8)
            mae = metrics['MPJAE_per_joint'][jname]
            pcc = metrics['PCC_per_joint'][jname]
            ax.set_title(f"{jname}\nMAE={mae:.1f} PCC={pcc:.2f}", fontsize=9)
            ax.grid(True, alpha=0.2)
            if i == 0:
                ax.legend(fontsize=7)

        self.fig_pred.tight_layout()
        self.canvas_pred.draw()

    def _export_csv(self):
        metrics = self.state.get('metrics')
        if metrics is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Metrics CSV", "metrics.csv", "CSV (*.csv)")
        if path:
            from core.evaluator import save_metrics_csv
            save_metrics_csv(metrics, str(Path(path).parent))
            QMessageBox.information(self, "Saved", f"Metrics saved to {path}")

    def _export_report(self):
        metrics = self.state.get('metrics')
        if metrics is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", "report.txt", "Text (*.txt)")
        if path:
            from core.evaluator import generate_report
            model = self.state.get('model')
            sel_sensors = self.state.get('selected_sensors', [])
            generate_report(metrics, str(Path(path).parent), model=model, sensor_names=sel_sensors)
            QMessageBox.information(self, "Saved", f"Report saved to {path}")

    def _save_plots(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        folder = Path(folder)
        self.fig_importance.savefig(folder / 'sensor_importance.png', dpi=150, bbox_inches='tight')
        self.fig_pred.savefig(folder / 'prediction_curves.png', dpi=150, bbox_inches='tight')
        QMessageBox.information(self, "Saved", f"Plots saved to {folder}")
