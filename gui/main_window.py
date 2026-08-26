"""
Main application window with 5-tab navigation.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QMenuBar, QMessageBox
)
from PyQt6.QtCore import Qt

from gui.panels.data_import import DataImportPanel
from gui.panels.skeleton_viewer import SkeletonViewerPanel
from gui.panels.angle_comparison import AngleComparisonPanel
from gui.panels.training import TrainingPanel
from gui.panels.results_dashboard import ResultsDashboardPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E-Textile Validation GUI")
        self.setMinimumSize(1200, 800)

        # Shared application state
        self.state = {
            # Subjects (populated by Tab 1). Each entry is a dict built by
            # make_subject(); one entry per recording session.
            'train_subjects': [],
            'test_subjects': [],
            # User selections (from Tab 3)
            'selected_angles': [],    # list of angle names
            'selected_sensors': [],   # list of sensor column names
            'alignment_method': 'upsample_sensor',
            # Training results (from Tab 4)
            'model': None,
            'train_losses': None,
            'predictions': None,      # pooled over test subjects
            'ground_truth': None,
            'metrics': None,          # pooled metrics dict
            'metrics_per_subject': None,  # {subject_id: metrics dict}
            'scaler': None,
            'config': None,
        }

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_import = DataImportPanel(self.state)
        self.tab_skeleton = SkeletonViewerPanel(self.state)
        self.tab_comparison = AngleComparisonPanel(self.state)
        self.tab_training = TrainingPanel(self.state)
        self.tab_results = ResultsDashboardPanel(self.state)

        self.tabs.addTab(self.tab_import, "1. Data Import")
        self.tabs.addTab(self.tab_skeleton, "2. Skeleton Viewer")
        self.tabs.addTab(self.tab_comparison, "3. Angles & Sensors")
        self.tabs.addTab(self.tab_training, "4. Training")
        self.tabs.addTab(self.tab_results, "5. Results")

        # Connect signals: when tab changes, refresh the new tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Status bar
        self.statusBar().showMessage("Ready. Start by importing data in Tab 1.")

    def _on_tab_changed(self, index):
        """Refresh the panel when user switches to it."""
        panel = self.tabs.widget(index)
        if hasattr(panel, 'refresh'):
            panel.refresh()
