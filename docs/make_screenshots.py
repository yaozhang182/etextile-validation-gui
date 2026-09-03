#!/usr/bin/env python3
"""
Generate the manual's screenshots.

Figures in a manual go stale the moment the UI changes, and hand-taken ones drift
apart from each other. This drives the real application headlessly, so after any
UI change one command regenerates every figure — and the manual, the README and
the paper's Fig. 2 are guaranteed to show the same build.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=. python docs/make_screenshots.py

Uses the bundled example sessions. Joint angles come from the on-disk cache when
present; otherwise the tabs that need a skeleton are captured in their empty
state, and a warning says so. To get the full set, run the extraction once in the
app (or via --extract) so the caches exist.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np                                          # noqa: E402
from PyQt6.QtWidgets import QApplication                    # noqa: E402

OUT = ROOT / 'docs' / 'img'
EXAMPLES = ROOT / 'input_data' / 'examples_data'
WINDOW = (1280, 840)

TARGET_ANGLES = ['right_shoulder_flexion', 'right_shoulder_abduction',
                 'right_elbow_flexion']


def add_session(window, split, subject_id, n):
    from gui.state import make_subject
    from gui.panels.data_import import read_csv_auto
    d = EXAMPLES / f'session_{n}'
    s = make_subject(subject_id, str(d / 'video.mp4'),
                     str(d / 'sensor.csv'), str(d / 'frames.csv'))
    s['sensor_df'] = read_csv_auto(s['sensor_path'])
    s['video_df'] = read_csv_auto(s['video_csv_path'])
    window.state[f'{split}_subjects'].append(s)
    window.tab_import._load_cached_angles(s)
    return s


def shoot(window, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.png'
    window.grab().save(str(path))
    print(f"  wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', action='store_true',
                    help='run pose extraction first (slow) to populate caches')
    ap.add_argument('--train', action='store_true',
                    help='train a short model so tabs 4 and 5 show real results')
    ap.add_argument('--epochs', type=int, default=3,
                    help='epochs for --train (default 3: enough for a figure)')
    args = ap.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    from gui.theme import apply_theme
    apply_theme(app)

    from gui.main_window import MainWindow
    from gui.state import ready_subjects

    window = MainWindow()
    window.resize(*WINDOW)
    window.show()

    subjects = [
        ('train', 'S1', 1), ('train', 'S2', 2),
        ('train', 'S3', 4), ('test', 'T1', 3),
    ]
    for split, sid, n in subjects:
        add_session(window, split, sid, n)

    if args.extract:
        print("Extracting skeletons (slow)...")
        window.tab_import._extract_all()
        while (window.tab_import._pending_extractions
               or any(s['status'] == 'extracting'
                      for k in ('train', 'test')
                      for s in window.state[f'{k}_subjects'])):
            app.processEvents()
            for w in window.tab_import._workers:
                w.wait(200)
            app.processEvents()

    have_skeletons = bool(ready_subjects(window.state, 'train'))
    if not have_skeletons:
        print("NOTE: no cached joint angles found — tabs 2-5 will be captured\n"
              "      in their empty state. Re-run with --extract for the full set.")

    # --- Tab 1 ---------------------------------------------------------
    window.tabs.setCurrentIndex(0)
    window.tab_import.refresh()
    window.tab_import.preview.setPlainText(
        "[train/S1] added\n"
        "  video: 454 frames @ 30.2 fps\n"
        "  sensor: 143 rows, columns ['EpochTime', 'S1'...'S10']\n"
        "  usable overlap: 12.9 s"
    )
    app.processEvents()
    shoot(window, 'tab1_data_import')

    # --- Tab 2 ---------------------------------------------------------
    window.tabs.setCurrentIndex(1)
    if have_skeletons:
        window.tab_skeleton.frame_slider.setValue(
            max(window.tab_skeleton.frame_slider.maximum() // 3, 0))
    app.processEvents()
    shoot(window, 'tab2_skeleton_viewer')

    # --- Tab 3 ---------------------------------------------------------
    window.tabs.setCurrentIndex(2)
    for name in TARGET_ANGLES:
        cb = window.tab_comparison.angle_checkboxes.get(name)
        if cb is not None:
            cb.setChecked(True)
    app.processEvents()
    shoot(window, 'tab3_angles_sensors')

    # --- Tab 4 ---------------------------------------------------------
    window.tabs.setCurrentIndex(3)
    window.tab_training.refresh()

    if args.train and have_skeletons:
        # A short run: the figures need a populated dashboard, not a good model.
        print(f"Training {args.epochs} epoch(s) to populate tabs 4 and 5...")
        t = window.tab_training
        t.epochs_spin.setValue(args.epochs)
        t.seq_spin.setValue(20)
        t.batch_spin.setValue(64)
        t._start_training()
        if t._worker is not None:
            while t._worker.isRunning():
                app.processEvents()
                t._worker.wait(200)
            app.processEvents()      # let the finished signal deliver
            for _ in range(50):
                app.processEvents()
        if window.state.get('metrics') is None:
            print("  WARNING: training produced no metrics; see the log pane")
    elif args.train:
        print("  --train needs cached angles; run --extract first")

    app.processEvents()
    shoot(window, 'tab4_training')

    # --- Tab 5 ---------------------------------------------------------
    window.tabs.setCurrentIndex(4)
    window.tab_results.refresh()
    app.processEvents()
    shoot(window, 'tab5_results')

    # --- Add Subject dialog, valid and blocked -------------------------
    from gui.panels.data_import import AddSubjectDialog
    d1 = EXAMPLES / 'session_1'
    dlg = AddSubjectDialog(window, 'train', 'S5')
    dlg.resize(700, 420)
    dlg.video_edit.setText(str(d1 / 'video.mp4'))
    dlg.sensor_edit.setText(str(d1 / 'sensor.csv'))
    dlg.times_edit.setText(str(d1 / 'frames.csv'))
    dlg._validate()
    dlg.show(); app.processEvents()
    shoot(dlg, 'dialog_add_subject')

    # The same dialog with the two CSVs swapped: shows how a problem is reported.
    dlg.sensor_edit.setText(str(d1 / 'frames.csv'))
    dlg.times_edit.setText(str(d1 / 'sensor.csv'))
    dlg._validate()
    app.processEvents()
    shoot(dlg, 'dialog_validation_error')
    dlg.close()

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
