"""
Tests for the Tab 3 video cursor (gui/panels/angle_comparison.py, video_scrubber.py).

Run: QT_QPA_PLATFORM=offscreen python tests/test_cursor.py
"""

import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import QApplication  # noqa: E402
_app = QApplication.instance() or QApplication(sys.argv)

from gui.main_window import MainWindow  # noqa: E402
from gui.panels.data_import import read_csv_auto  # noqa: E402
from gui.state import make_subject  # noqa: E402

SESSION = os.path.join(ROOT, 'input_data', 'examples_data', 'session_1')
ANGLES = ['right_shoulder_flexion', 'right_elbow_flexion']


def _window():
    """Main window on Tab 3 with example session 1 loaded and two angles ticked."""
    w = MainWindow()
    w.resize(1280, 900)
    w.show()
    s = make_subject('S1', os.path.join(SESSION, 'video.mp4'),
                     os.path.join(SESSION, 'sensor.csv'), os.path.join(SESSION, 'frames.csv'))
    s['sensor_df'] = read_csv_auto(s['sensor_path'])
    s['video_df'] = read_csv_auto(s['video_csv_path'])
    w.state['train_subjects'].append(s)
    w.tab_import._load_cached_angles(s)
    assert s['status'] == 'ready', "example session 1 needs its angle cache"
    tab = w.tab_comparison
    w.tabs.setCurrentWidget(tab)
    _app.processEvents()
    for name in ANGLES:
        tab.angle_checkboxes[name].setChecked(True)
    _app.processEvents()
    return w, tab, s


def _shown_values(tab):
    """{channel: value string} as displayed above the plots."""
    text = re.sub(r'<[^>]+>', ' ', tab.values_label.text()).replace('&nbsp;', ' ')
    text = text.replace('&#9632;', ' ')
    return dict(re.findall(r'(\S+)\s+(—|-?\d+\.\d)', text))


def test_cursor_follows_the_video_and_shows_values():
    w, tab, s = _window()
    tab.scrubber.set_frame(200)
    _app.processEvents()
    c = tab._cursor
    assert c is not None
    t = c['angle_t'][200]
    for line in c['lines']:
        assert np.allclose(line.get_xdata(), [t, t])
    assert len(c['lines']) == 2, "aligned streams get a line in both panels"

    shown = _shown_values(tab)
    for name in ANGLES:
        assert shown[name].rstrip('°') == f"{s['angles'][name][200]:.1f}", name
    # sensor value: linear interpolation at that instant on the shared clock
    name, x, vals, _ = c['sensors'][0]
    assert shown[name] == f"{np.interp(t, x, vals):.1f}"
    assert 'Frame 200' in tab.cursor_label.text()


def test_sensor_shows_a_dash_outside_its_recording():
    w, tab, s = _window()
    c = tab._cursor
    x = c['sensors'][0][1]
    late = np.flatnonzero(c['angle_t'] > x[-1])
    assert late.size, "session 1's video runs past the sensor; the test relies on it"
    tab.scrubber.set_frame(int(late[0]))
    _app.processEvents()
    assert _shown_values(tab)[c['sensors'][0][0]] == '—'


def test_clicking_a_plot_jumps_the_video():
    from matplotlib.backend_bases import MouseEvent
    w, tab, s = _window()
    t = tab._cursor['angle_t']
    for ax in (tab.ax_angle, tab.ax_sensor):
        x, y = ax.transData.transform((t[321], np.mean(ax.get_ylim())))
        tab.canvas.callbacks.process('button_press_event',
                                     MouseEvent('button_press_event', tab.canvas, x, y, button=1))
        _app.processEvents()
        assert abs(tab.scrubber.frame() - 321) <= 1, tab.scrubber.frame()
        tab.scrubber.set_frame(0)


def test_frame_is_shared_with_the_skeleton_viewer():
    w, tab, s = _window()
    tab.scrubber.set_frame(150)
    _app.processEvents()
    assert s['view_frame'] == 150
    w.tabs.setCurrentWidget(w.tab_skeleton)
    _app.processEvents()
    assert w.tab_skeleton.frame_slider.value() == 150
    w.tab_skeleton.frame_slider.setValue(60)
    w.tabs.setCurrentWidget(tab)
    _app.processEvents()
    assert tab.scrubber.frame() == 60


def test_moving_the_cursor_does_not_redraw_the_plots():
    """The cursor is blitted; a full redraw per frame (~100 ms) would stall playback."""
    w, tab, s = _window()
    draws = []
    tab.canvas.mpl_connect('draw_event', lambda e: draws.append(1))
    for k in range(10, 40):
        tab.scrubber.set_frame(k)
        _app.processEvents()
    assert not draws, f"{len(draws)} full redraws"


def test_playback_keeps_real_time_by_skipping_frames():
    w, tab, s = _window()
    sc = tab.scrubber
    sc.set_frame(0)
    sc.toggle_play()
    sc._play_start = (time.perf_counter() - 2.0, 0)   # pretend 2 s have passed
    sc._advance()
    expected = int(2.0 * sc._fps)
    assert abs(sc.frame() - expected) <= 2, (sc.frame(), expected)
    sc.stop()
    assert not sc.playing


def test_hiding_the_tab_pauses_playback():
    w, tab, s = _window()
    tab.scrubber.toggle_play()
    assert tab.scrubber.playing
    w.tabs.setCurrentWidget(w.tab_skeleton)
    _app.processEvents()
    assert not tab.scrubber.playing


def test_no_angles_means_no_cursor():
    w = MainWindow()
    tab = w.tab_comparison
    w.tabs.setCurrentWidget(tab)
    _app.processEvents()
    tab.scrubber.set_frame(5)
    assert tab._cursor is None and tab.values_label.text() == ''


def main():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
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
