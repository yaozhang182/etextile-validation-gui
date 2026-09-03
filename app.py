#!/usr/bin/env python3
"""
E-Textile Validation GUI — Main entry point.

Usage:
    python app.py
    python app.py --version     # print version and exit
    python app.py --selftest    # exercise the pipeline headlessly and exit
"""

import sys

__version__ = '0.2.2'


def selftest():
    """
    Exercise the real pipeline without a display.

    Run by CI against the packaged Windows binary. A packaged app can build
    cleanly and still fail on first use because a dependency was pruned — the
    0.2.0 build died with "cannot import name 'distributions' from partially
    initialized module 'torch'" because a torch submodule had been excluded.
    Importing the heavy dependencies and running one training step is what
    actually catches that, so this does both.
    """
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failures.append(name)

    print(f"E-Textile Validation GUI {__version__} — self-test")

    # Third-party imports, individually so a failure names the culprit.
    def import_torch():
        import torch
        torch.distributions      # eagerly imported by torch/__init__; see docstring
        torch.nn
    check('import torch', import_torch)
    check('import mediapipe', lambda: __import__('mediapipe'))
    check('import cv2', lambda: __import__('cv2'))
    check('import sklearn', lambda: __import__('sklearn.preprocessing'))
    check('import scipy', lambda: __import__('scipy.interpolate'))
    check('import pandas', lambda: __import__('pandas'))
    check('import matplotlib qt backend',
          lambda: __import__('matplotlib.backends.backend_qtagg'))
    check('import seaborn', lambda: __import__('seaborn'))
    check('import PyQt6', lambda: __import__('PyQt6.QtWidgets'))

    # MediaPipe's model assets are data files, not imports — PyInstaller misses
    # them unless collected, and the app then crashes only on first real use.
    def mediapipe_models():
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0)
        pose.close()
    check('mediapipe pose model loads', mediapipe_models)

    # Application modules.
    check('import core modules', lambda: [
        __import__(m) for m in ('core.smpl_extraction', 'core.joint_angles',
                                'core.data_alignment', 'core.model',
                                'core.trainer', 'core.evaluator',
                                'core.confidence')])
    check('import gui modules', lambda: __import__('gui.main_window'))

    # One real training step — the path that failed in 0.2.0.
    def train_one_step():
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from core.data_alignment import ShoulderDataset
        from core.model import create_model
        from core.trainer import train_model, evaluate_model
        from core.evaluator import calculate_metrics

        rng = np.random.default_rng(0)
        X = rng.random((80, 4)).astype('float32')
        Y = rng.random((80, 2)).astype('float32')
        cfg = {'model_type': 'hybrid_cnn_lstm', 'target_cols': ['a', 'b'],
               'sequence_length': 10, 'batch_size': 8, 'learning_rate': 1e-3,
               'epochs': 1, 'cnn_filters': 8, 'lstm_hidden': 8, 'dropout': 0.1}
        ds = ShoulderDataset(X, Y, cfg['sequence_length'])
        model = create_model(cfg, num_sensors=4)
        model, _ = train_model(model, DataLoader(ds, batch_size=8, shuffle=True),
                               cfg, 'cpu')
        preds, gt = evaluate_model(model, DataLoader(ds, batch_size=16), 'cpu')
        calculate_metrics(gt, preds, ['a', 'b'])
    check('train + evaluate one step', train_one_step)

    # The window must actually construct offscreen.
    def build_window():
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import MainWindow
        from gui.theme import apply_theme, build_stylesheet
        app = QApplication.instance() or QApplication([])
        # A stylesheet that fails to load is a packaging bug: style.qss is a
        # data file and PyInstaller drops it unless it is in the spec's datas.
        assert build_stylesheet(), "style.qss missing from the bundle"
        from gui import icons
        assert icons.available(), "gui/icons missing from the bundle"
        apply_theme(app)
        w = MainWindow()
        for tab in (w.tab_import, w.tab_skeleton, w.tab_comparison,
                    w.tab_training, w.tab_results):
            tab.refresh()
    check('build main window', build_window)

    if failures:
        print(f"\nSELF-TEST FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("\nSELF-TEST PASSED")
    return 0


def main():
    # Handled before Qt starts so the packaged .exe can be checked without a display.
    if '--version' in sys.argv:
        print(f"E-Textile Validation GUI {__version__}")
        return 0
    if '--selftest' in sys.argv:
        return selftest()

    from PyQt6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    from gui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("E-Textile Validation GUI")
    app.setApplicationVersion(__version__)
    apply_theme(app)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
