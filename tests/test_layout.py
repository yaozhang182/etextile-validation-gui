#!/usr/bin/env python3
"""
Layout regression tests: the controls must actually be on screen.

Every other test here asserts on dictionaries, labels and properties — which is
why a real failure shipped unnoticed. Tab 3's joint-angle list lived in a
QScrollArea whose minimum size hint is a few pixels, so when the sensor column
below it grew, Qt squeezed the scroll area down to its scrollbar: the widgets
still existed, still answered isChecked(), and were simply not visible. No
assertion about state can catch that. These assertions are about geometry.

Run headless:

    QT_QPA_PLATFORM=offscreen PYTHONPATH=. python tests/test_layout.py
"""

import os
import sys
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from gui.main_window import MainWindow  # noqa: E402
from test_multisubject import fake_subject  # noqa: E402

# 1280x800 is the smallest laptop the workshop is likely to be run on; the rest
# bracket it. The bug reproduced at ordinary sizes, not just tiny ones.
WINDOW_SIZES = [(1280, 800), (1440, 900), (1920, 1080)]

# Ten channels is the example garment; 24 is a plausible full sleeve, and the
# case where an unbounded checkbox column does the most damage.
CHANNEL_COUNTS = [4, 10, 24]


def _window(n_channels, size):
    """
    A window in the state the app really reaches Tab 3 in.

    The order matters and is the order the application uses: the window is on
    screen, data arrives while Tab 1 is still in front, and only then does the
    user switch. Building the panel before show() hides a whole class of bug —
    a widget laid out while its tab is hidden is exactly the case that broke.
    """
    cols = tuple(f'S{i + 1}' for i in range(n_channels))
    w = MainWindow()
    w.resize(*size)
    w.show()
    _app.processEvents()

    w.state['train_subjects'].append(fake_subject('S1', 200, 400, 1e9, cols=cols))
    w.tab_comparison.refresh()          # still on Tab 1 at this point
    _app.processEvents()

    w.tabs.setCurrentWidget(w.tab_comparison)
    _app.processEvents()
    return w


def test_angle_list_is_visible_at_every_window_size():
    """The reported bug: the scroll area collapsed to its scrollbar."""
    for size in WINDOW_SIZES:
        for n in CHANNEL_COUNTS:
            w = _window(n, size)
            panel = w.tab_comparison
            h = panel.angle_scroll.viewport().height()
            assert h >= 150, \
                f"angle list viewport only {h}px at {size} with {n} channels"

            # Not just a tall viewport — actual checkboxes drawn inside it.
            shown = [cb for cb in panel.angle_checkboxes.values()
                     if not cb.visibleRegion().isEmpty()]
            assert len(shown) >= 4, \
                f"only {len(shown)} angle checkboxes visible at {size}/{n}ch"
            w.close()


def test_sensor_list_is_visible_and_bounded():
    """Sensors must stay visible, and must not grow without limit."""
    for size in WINDOW_SIZES:
        for n in CHANNEL_COUNTS:
            w = _window(n, size)
            panel = w.tab_comparison
            assert panel.sensor_scroll.viewport().height() >= 100, size
            shown = [cb for cb in panel.sensor_checkboxes.values()
                     if not cb.visibleRegion().isEmpty()]
            assert shown, f"no sensor checkboxes visible at {size} with {n}ch"

            # The whole point: the sensor group cannot claim the column.
            left = panel.angle_scroll.parentWidget()
            assert panel.sensor_group.height() < left.window().height(), size
            w.close()


def test_list_rows_are_never_squeezed_below_their_own_height():
    """
    A scroll area must scroll, not compress.

    Rebuilding the sensor list while Tab 3 was hidden left the inner widget at
    the viewport height, and its layout then shrank every checkbox to about half
    its natural size — the labels rendered as unreadable stubs. The rows still
    existed and still answered isChecked(), so only their geometry shows it.
    """
    for n in CHANNEL_COUNTS:
        for panel_attr, boxes_attr in (('sensor_scroll', 'sensor_checkboxes'),
                                       ('angle_scroll', 'angle_checkboxes')):
            w = _window(n, (1280, 840))
            panel = w.tab_comparison
            boxes = getattr(panel, boxes_attr).values()
            for cb in boxes:
                natural = cb.minimumSizeHint().height()
                assert cb.height() >= natural, (
                    f"{boxes_attr} row {cb.text()!r} drawn {cb.height()}px "
                    f"but needs {natural}px ({n} channels)")
            # ...which is only possible because the content is allowed to
            # overflow and scroll.
            scroll = getattr(panel, panel_attr)
            assert scroll.widget().minimumHeight() > 0, panel_attr
            w.close()


def test_panel_minimum_height_does_not_grow_with_channels():
    """
    The root cause, asserted directly.

    With an unbounded checkbox column the tab's minimum height rose with the
    channel count — 24 channels demanded a 978px tab, so on a 768px laptop the
    window could not shrink to fit and the compressible widget (the angle scroll
    area) was crushed to its scrollbar. Inside scroll areas the minimum is flat.
    """
    heights = []
    for n in CHANNEL_COUNTS:
        w = _window(n, (1280, 800))
        heights.append(w.tab_comparison.minimumSizeHint().height())
        w.close()
    assert max(heights) - min(heights) == 0, \
        f"minimum height varies with channel count: {dict(zip(CHANNEL_COUNTS, heights))}"
    # Must fit a 1366x768 laptop with room for title bar and taskbar.
    assert max(heights) <= 700, f"tab needs {max(heights)}px minimum"


def test_both_lists_share_the_column():
    """Neither group may starve the other, whatever the channel count."""
    for n in CHANNEL_COUNTS:
        w = _window(n, (1280, 800))
        panel = w.tab_comparison
        angle_h = panel.angle_scroll.viewport().height()
        sensor_h = panel.sensor_scroll.viewport().height()
        # Angles are the longer list (26 channels), so they get the larger share.
        assert angle_h >= sensor_h, (n, angle_h, sensor_h)
        w.close()


def test_angle_labels_fit_the_panel_width():
    """Confidence suffixes lengthened every label; the panel must still hold them."""
    import numpy as np
    w = MainWindow()
    w.resize(1280, 800)
    s = fake_subject('S1', 200, 400, 1e9)
    s['skeleton']['visibility'] = np.full((400, 33), 0.93, dtype=np.float32)
    w.state['train_subjects'].append(s)
    w.tab_comparison.refresh()
    w.tabs.setCurrentWidget(w.tab_comparison)
    w.show()
    _app.processEvents()

    panel = w.tab_comparison
    widest = max(cb.sizeHint().width() for cb in panel.angle_checkboxes.values())
    # A horizontal scrollbar for one long label is survivable; being unable to
    # read most of the list is not.
    assert widest <= panel.angle_scroll.viewport().width() + 40, \
        f"widest angle label {widest}px vs {panel.angle_scroll.viewport().width()}px viewport"
    w.close()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
