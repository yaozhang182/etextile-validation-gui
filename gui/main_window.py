"""
Main application window with 5-tab navigation.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QMenuBar, QMessageBox, QPushButton
)
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QAction, QDesktopServices

from gui.panels.data_import import DataImportPanel
from gui.panels.skeleton_viewer import SkeletonViewerPanel
from gui.panels.angle_comparison import AngleComparisonPanel
from gui.panels.training import TrainingPanel
from gui.panels.results_dashboard import ResultsDashboardPanel
from gui.state import ready_subjects

#: Where the manual lives. Each tab deep-links to its own section, so help is
#: contextual rather than dumping the reader at the top of a long page.
MANUAL_URL = ("https://github.com/yaozhang182/etextile-validation-gui"
              "/blob/main/docs/manual.md")

#: Tab index -> (title, manual anchor). The anchors match the headings in
#: docs/manual.md; keep them in step.
TABS = [
    ("1  Data Import",      "#tab-1--data-import"),
    ("2  Skeleton Viewer",  "#tab-2--skeleton-viewer"),
    ("3  Angles && Sensors", "#tab-3--angles--sensors"),
    ("4  Training",         "#tab-4--training"),
    ("5  Results",          "#tab-5--results"),
]


def manual_url(anchor=""):
    """Full URL of the manual, optionally at a specific section."""
    return MANUAL_URL + anchor


def open_manual(anchor=""):
    """Open the manual in the user's browser."""
    QDesktopServices.openUrl(QUrl(manual_url(anchor)))


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
        self.tabs.setDocumentMode(True)   # flat tab rail, styled in style.qss
        self.setCentralWidget(self.tabs)

        self.tab_import = DataImportPanel(self.state)
        self.tab_skeleton = SkeletonViewerPanel(self.state)
        self.tab_comparison = AngleComparisonPanel(self.state)
        self.tab_training = TrainingPanel(self.state)
        self.tab_results = ResultsDashboardPanel(self.state)

        for panel, (title, _) in zip(
            (self.tab_import, self.tab_skeleton, self.tab_comparison,
             self.tab_training, self.tab_results), TABS,
        ):
            self.tabs.addTab(panel, title)

        self.help_btn = QPushButton("?")
        self.help_btn.setProperty("subtle", True)
        self.help_btn.setFixedSize(26, 24)
        self.help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_btn.clicked.connect(
            lambda: open_manual(TABS[self.tabs.currentIndex()][1])
        )
        self.tabs.setCornerWidget(self.help_btn, Qt.Corner.TopRightCorner)

        # Connect signals: when tab changes, refresh the new tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._build_menu()

        # Status bar
        self.statusBar().showMessage("Ready. Start by importing data in Tab 1.")
        self._update_tab_states()
        # currentChanged does not fire for the tab already selected at startup.
        self._on_tab_changed(self.tabs.currentIndex())

        # State is mutated by the panels, which have no reference back here, so
        # poll instead of wiring a signal through every one of them. The check is
        # five dict lookups; the cost is irrelevant next to the plots.
        self._state_poll = QTimer(self)
        self._state_poll.setInterval(600)
        self._state_poll.timeout.connect(self._update_tab_states)
        self._state_poll.start()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        help_menu = self.menuBar().addMenu("&Help")

        manual = QAction("User &Manual", self)
        manual.setShortcut("F1")
        manual.setStatusTip("Open the full manual on GitHub")
        manual.triggered.connect(lambda: open_manual())
        help_menu.addAction(manual)

        current = QAction("Help for &This Tab", self)
        current.setStatusTip("Open the manual at the section for the current tab")
        current.triggered.connect(
            lambda: open_manual(TABS[self.tabs.currentIndex()][1])
        )
        help_menu.addAction(current)

        help_menu.addSeparator()

        data_help = QAction("Input Data &Format", self)
        data_help.setStatusTip("What files the tool needs and how they must look")
        data_help.triggered.connect(lambda: open_manual("#input-requirements"))
        help_menu.addAction(data_help)

        trouble = QAction("&Troubleshooting", self)
        trouble.triggered.connect(lambda: open_manual("#troubleshooting"))
        help_menu.addAction(trouble)

        help_menu.addSeparator()

        about = QAction("&About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _show_about(self):
        from app import __version__
        QMessageBox.about(
            self, "About",
            f"<b>E-Textile Validation GUI</b> {__version__}"
            "<p>Validate a wearable motion-sensing garment from a webcam "
            "recording and a sensor CSV — no motion capture, no code.</p>"
            f'<p><a href="{manual_url()}">User manual</a></p>'
        )

    # ------------------------------------------------------------------
    # Workflow state
    # ------------------------------------------------------------------

    def _workflow_progress(self):
        """
        How far the user has got. The five tabs are a strict sequence, so the
        tab bar can show what is ready and what is still waiting on an earlier
        step. Returns the number of steps completed.
        """
        s = self.state
        if s.get('metrics') is not None:
            return 5
        if s.get('model') is not None:
            return 4
        if s.get('selected_angles') and s.get('selected_sensors'):
            return 3
        if ready_subjects(s, 'train') or ready_subjects(s, 'test'):
            return 2
        if s.get('train_subjects') or s.get('test_subjects'):
            return 1
        return 0

    def _update_tab_states(self):
        """
        Show which steps are done in the tab bar.

        Every tab stays *enabled* on purpose. Disabling the later ones would
        deadlock the user: adding a subject in Tab 1 does not itself fire a tab
        change, so a tab disabled at startup would never be re-enabled and could
        never be clicked. Each panel already explains its own empty state, so a
        tick mark is enough to convey progress.
        """
        done = self._workflow_progress()
        needs = [
            "",
            "Needs a subject added in Tab 1",
            "Needs a skeleton extracted in Tab 1",
            "Needs angle and sensor channels chosen in Tab 3",
            "Needs a trained model from Tab 4",
        ]
        for index, (title, _) in enumerate(TABS):
            complete = done > index
            self.tabs.setTabText(index, f"{title}  \u2713" if complete else title)
            self.tabs.setTabToolTip(
                index, "" if complete or index == 0 else needs[index]
            )

    def _on_tab_changed(self, index):
        """Refresh the panel when user switches to it."""
        self._update_tab_states()
        title = TABS[index][0].split('  ', 1)[-1].replace('&&', '&')
        self.help_btn.setToolTip(f"Open the manual at \u201c{title}\u201d")
        panel = self.tabs.widget(index)
        if hasattr(panel, 'refresh'):
            panel.refresh()
