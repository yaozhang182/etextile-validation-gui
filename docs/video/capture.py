"""
Drive the real application headlessly and capture every screen state the
tutorial needs, at 1920x1080, together with the pixel rectangle of each control
that the video will point at. No state is faked: sessions are imported,
extracted and trained for real on the bundled example data.
"""
import json, os, shutil, sys, time
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
BUILD = os.environ.get('VIDEO_BUILD', os.path.join(ROOT, 'docs', 'video', 'build'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['QT_SCALE_FACTOR'] = '1.3333'   # a 1440x810 window renders as native 1920x1080
# PyTorch otherwise starts one thread per core; on a shared machine with a small
# CPU quota that oversubscribes and training slows to a crawl.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
OUT = os.path.join(BUILD, 'shots'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT); os.chdir(ROOT)

import numpy as np
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton, QScrollArea

app = QApplication(sys.argv)
from gui.theme import apply_theme; apply_theme(app)
from gui.main_window import MainWindow
from gui.state import make_subject
from gui.panels.data_import import read_csv_auto, AddSubjectDialog, angle_cache_path
from gui.help import HelpBadge

S = 1.3333
SHOTS = {}

def pump(n=6):
    for _ in range(n):
        app.processEvents()

def img_of(widget):
    return widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)

def rect_in(widget, root, offset=(0, 0)):
    p = widget.mapTo(root, QPoint(0, 0))
    return [round(p.x() * S) + offset[0], round(p.y() * S) + offset[1],
            round(widget.width() * S), round(widget.height() * S)]

def overlay(base, top, pos=None, dim=True):
    """Paint a dialog onto the window image, centred unless pos is given."""
    # Work in physical pixels. The grabbed images carry the 1.3333 device pixel
    # ratio, and QPainter would otherwise scale every position and size by it.
    out = QImage(base); out.setDevicePixelRatio(1.0)
    top = QImage(top); top.setDevicePixelRatio(1.0)
    p = QPainter(out)
    if dim:
        p.fillRect(out.rect(), QColor(20, 28, 38, 90))
    x = (out.width() - top.width()) // 2 if pos is None else pos[0]
    y = (out.height() - top.height()) // 2 if pos is None else pos[1]
    for k in range(10, 0, -2):                                    # soft shadow
        p.fillRect(QRect(x - k + 6, y - k + 10, top.width() + 2 * k, top.height() + 2 * k),
                   QColor(0, 0, 0, 10))
    p.drawImage(x, y, top)
    p.setPen(QColor(150, 160, 172)); p.drawRect(x - 1, y - 1, top.width() + 1, top.height() + 1)
    p.end()
    return out, (x, y)

def save(name, image, rects=None):
    path = os.path.join(OUT, name + '.png')
    image.save(path)
    SHOTS[name] = {'file': path, 'rects': rects or {}}
    print(f'  shot {name}', flush=True)

def button(root, text):
    for b in root.findChildren(QPushButton):
        if b.isVisible() and text.lower() in b.text().replace('&', '').lower():
            return b
    raise KeyError(text)

def add_session(w, split, sid, n):
    d = f'input_data/examples_data/session_{n}'
    s = make_subject(sid, f'{d}/video.mp4', f'{d}/sensor.csv', f'{d}/frames.csv')
    s['sensor_df'] = read_csv_auto(s['sensor_path']); s['video_df'] = read_csv_auto(s['video_csv_path'])
    w.state[f'{split}_subjects'].append(s)
    w.tab_import._load_cached_angles(s)
    return s

# Blocking dialogs are grabbed instead of run.
GRABBED = {}
def fake_exec(self):
    self.show(); pump(8); GRABBED['last'] = (img_of(self), self)
    self.hide(); return 0
QDialog.exec = fake_exec
QMessageBox.exec = fake_exec

w = MainWindow(); w.resize(1440, 810); w.show(); pump(10)
win = lambda: img_of(w)

# ---------------------------------------------------------------- Tab 1
t1 = w.tab_import
mb = w.menuBar()
bar = {a.text(): a for a in mb.actions()}
def mb_rect(text):
    g = mb.actionGeometry(bar[text]); o = mb.mapTo(w, QPoint(0, 0))
    return [round((o.x() + g.x()) * S), round((o.y() + g.y()) * S), round(g.width() * S), round(g.height() * S)]

save('tab1_empty', win(), {'tabbar': rect_in(w.tabs.tabBar(), w), 'extract': rect_in(t1.extract_btn, w)})

# the two Add Subject buttons, in table order
adds = [b for b in t1.findChildren(QPushButton) if 'Add Subject' in b.text()]
adds.sort(key=lambda b: b.mapTo(w, QPoint(0, 0)).y())
removes = [b for b in t1.findChildren(QPushButton) if 'Remove' in b.text()]
removes.sort(key=lambda b: b.mapTo(w, QPoint(0, 0)).y())
SHOTS['tab1_empty']['rects'].update({'add_train': rect_in(adds[0], w), 'add_test': rect_in(adds[1], w)})

base = win()
dlg = AddSubjectDialog(w, 'train', 'S1'); dlg.resize(760, 430); dlg.show(); pump(8)
im, off = overlay(base, img_of(dlg))
browse = sorted([b for b in dlg.findChildren(QPushButton) if 'Browse' in b.text()],
                key=lambda b: b.mapTo(dlg, QPoint(0, 0)).y())
save('dialog_empty', im, {'dialog': [off[0], off[1], round(dlg.width()*S), round(dlg.height()*S)],
                          'browse': [rect_in(b, dlg, off) for b in browse]})
d1 = 'input_data/examples_data/session_1'
dlg.video_edit.setText(f'{d1}/video.mp4'); dlg.sensor_edit.setText(f'{d1}/sensor.csv')
dlg.times_edit.setText(f'{d1}/frames.csv'); dlg._validate(); pump(8)
im, off = overlay(base, img_of(dlg))
ok = dlg.buttons.button(dlg.buttons.StandardButton.Ok)
save('dialog_filled', im, {'dialog': [off[0], off[1], round(dlg.width()*S), round(dlg.height()*S)],
                           'fields': rect_in(dlg.video_edit, dlg, off)[:2] + [rect_in(browse[0], dlg, off)[0] + rect_in(browse[0], dlg, off)[2] - rect_in(dlg.video_edit, dlg, off)[0], rect_in(dlg.times_edit, dlg, off)[1] + rect_in(dlg.times_edit, dlg, off)[3] - rect_in(dlg.video_edit, dlg, off)[1]],
                           'report': rect_in(dlg.report, dlg, off), 'ok': rect_in(ok, dlg, off)})
dlg.sensor_edit.setText(f'{d1}/frames.csv'); dlg.times_edit.setText(f'{d1}/sensor.csv'); dlg._validate(); pump(8)
im, off = overlay(base, img_of(dlg))
save('dialog_swapped', im, {'report': rect_in(dlg.report, dlg, off), 'ok': rect_in(ok, dlg, off)})
dlg.close()

# Session 3 is imported without its cache so that extraction really runs.
cache3 = angle_cache_path(f'{ROOT}/input_data/examples_data/session_3/video.mp4')
parked = str(cache3) + '.parked'
shutil.move(str(cache3), parked)
try:
    add_session(w, 'train', 'S1', 1); add_session(w, 'train', 'S2', 2); add_session(w, 'train', 'S3', 4)
    add_session(w, 'test', 'T1', 3)
    t1.refresh(); pump(10)
    save('tab1_tables', win(), {'train_table': rect_in(t1.tables['train'], w), 'test_table': rect_in(t1.tables['test'], w),
                                'remove_train': rect_in(removes[0], w), 'extract': rect_in(t1.extract_btn, w)})

    t1._extract_all(); pump(4)
    got = False
    t0 = time.time()
    while any(wk.isRunning() for wk in t1._workers) or t1._pending_extractions:
        pump(2)
        dg = t1._dialog
        if not got and dg is not None and dg.bar.maximum() > 0 and dg.bar.value() > 0.42 * dg.bar.maximum() \
                and time.time() - t0 > 3:
            base = win(); im, off = overlay(base, img_of(dg))
            cancel = dg.cancel_btn
            save('extract_progress', im, {'dialog': [off[0], off[1], round(dg.width()*S), round(dg.height()*S)],
                                          'bar': rect_in(dg.bar, dg, off), 'cancel': rect_in(cancel, dg, off)})
            got = True
        for wk in t1._workers:
            wk.wait(50)
    pump(30)
finally:
    # put the original cache back, exactly as it was
    if os.path.exists(parked):
        shutil.move(parked, str(cache3))
save('tab1_done', win(), {'test_table': rect_in(t1.tables['test'], w), 'log': rect_in(t1.preview, w)})

# ---------------------------------------------------------------- Tab 2
w.tabs.setCurrentIndex(1); pump(10)
t2 = w.tab_skeleton
tq = button(t2, 'Tracking Quality')
frames = list(range(96, 236, 4))
for i, f in enumerate(frames):
    t2.frame_slider.setValue(f); pump(4)
    save(f'tab2_f{i:02d}', win(), {'play': rect_in(t2.play_btn, w), 'slider': rect_in(t2.frame_slider, w),
                                   'tq': rect_in(tq, w)})
t2.frame_slider.setValue(160); pump(6)
cv = [c for c in t2.findChildren(type(w.tab_comparison.canvas))]
cv.sort(key=lambda c: c.mapTo(w, QPoint(0, 0)).x())
save('tab2_still', win(), {'video': rect_in(cv[0], w) if len(cv) > 1 else [0,0,0,0],
                           'skeleton': rect_in(cv[-1], w), 'tq': rect_in(tq, w)})
base = win()
tq.click(); pump(6)
qim, qdlg = GRABBED['last']
im, off = overlay(base, qim)
save('tab2_tracking', im, {'dialog': [off[0], off[1], qim.width(), qim.height()]})

# ---------------------------------------------------------------- Tab 3
w.tabs.setCurrentIndex(2); pump(10)
t3 = w.tab_comparison
t3.angle_scroll.verticalScrollBar().setValue(0); pump(4)
save('tab3_none', win(), {'angles': rect_in(t3.angle_scroll, w)})
targets = ['right_shoulder_flexion', 'right_shoulder_abduction', 'right_shoulder_rotation']
for n in targets:
    t3.angle_checkboxes[n].setChecked(True)
pump(6)
t3.angle_scroll.ensureWidgetVisible(t3.angle_checkboxes[targets[0]]); pump(6)
quick = [button(t3, 'Shoulder')]
all_btns = sorted([b for b in t3.findChildren(QPushButton) if b.text() in ('All', 'None', 'Shoulder')],
                  key=lambda b: (b.mapTo(w, QPoint(0, 0)).y(), b.mapTo(w, QPoint(0, 0)).x()))
angle_quick = [b for b in all_btns if b.mapTo(w, QPoint(0,0)).y() < t3.sensor_scroll.mapTo(w, QPoint(0,0)).y()]
qx = [rect_in(b, w) for b in angle_quick]
save('tab3_selected', win(), {
    'angles': rect_in(t3.angle_scroll, w),
    'target_box': rect_in(t3.angle_checkboxes[targets[0]], w),
    'quick': [min(r[0] for r in qx), min(r[1] for r in qx), max(r[0]+r[2] for r in qx) - min(r[0] for r in qx), max(r[3] for r in qx)],
    'sensors': rect_in(t3.sensor_scroll, w), 'sensor_buttons': rect_in(t3.sensor_buttons, w),
    'plots': rect_in(t3.canvas, w), 'info': rect_in(t3.info_label, w)})

# ---------------------------------------------------------------- Tab 4
w.tabs.setCurrentIndex(3); pump(10)
t4 = w.tab_training; t4.refresh(); pump(6)
badges = [b for b in t4.findChildren(HelpBadge) if b.isVisible()]
seq_badge = min(badges, key=lambda b: abs(b.mapTo(w, QPoint(0,0)).y() - t4.seq_spin.mapTo(w, QPoint(0,0)).y()))
hp = t4.model_combo.parentWidget()
save('tab4_ready', win(), {'summary': rect_in(t4.selection_label, w), 'hyper': rect_in(hp, w),
                           'start': rect_in(t4.train_btn, w), 'badge': rect_in(seq_badge, w)})

def train_and_capture(prefix):
    t4._start_training(); pump(4)
    got = False
    while t4._worker is not None and t4._worker.isRunning():
        pump(2)
        dg = t4._dialog
        if not got and dg is not None and dg.bar.maximum() > 0 and dg.bar.value() >= 0.4 * dg.bar.maximum():
            base = win(); im, off = overlay(base, img_of(dg))
            save(prefix + '_progress', im, {'dialog': [off[0], off[1], round(dg.width()*S), round(dg.height()*S)]})
            got = True
        t4._worker.wait(50)
    pump(40)

train_and_capture('train1')
loss = [c for c in t4.findChildren(type(t3.canvas)) if c.isVisible()]
save('tab4_done', win(), {'loss': rect_in(loss[0], w) if loss else [0,0,0,0]})

# ---------------------------------------------------------------- Tab 5
def results(prefix):
    w.tabs.setCurrentIndex(4); pump(12)
    t5 = w.tab_results
    sa = t5.findChild(QScrollArea); sa.verticalScrollBar().setValue(0); pump(6)
    save(prefix + '_top', win(), {'headline': rect_in(t5.info_label, w), 'table': rect_in(t5.metrics_table, w),
                                  'pred': rect_in(t5.canvas_pred, w)})
    # the confidence column: last visible column of the metrics table
    tbl = t5.metrics_table
    c = tbl.columnCount() - 1
    x = tbl.columnViewportPosition(c); vp = tbl.viewport().mapTo(w, QPoint(0, 0))
    SHOTS[prefix + '_top']['rects']['conf_col'] = [round((vp.x() + x) * S), round((tbl.mapTo(w, QPoint(0,0)).y()) * S),
                                                   round(tbl.columnWidth(c) * S), round(tbl.height() * S)]
    sa.verticalScrollBar().setValue(sa.verticalScrollBar().maximum()); pump(8)
    exports = [button(t5, 'Export Metrics'), button(t5, 'Export Report'), button(t5, 'Save All Plots')]
    xs = [rect_in(b, w) for b in exports]
    save(prefix + '_bottom', win(), {'importance': rect_in(t5.canvas_importance, w),
                                     'exports': [xs[0][0], xs[0][1], xs[2][0] + xs[2][2] - xs[0][0], xs[0][3]]})
    sa.verticalScrollBar().setValue(0); pump(4)

results('tab5')

# ---------------------------------------------------------------- iterate
imp = w.state['model'].get_feature_importance()
sens = list(w.state['selected_sensors'])
keep = [sens[i] for i in np.argsort(imp)[::-1][:5]]
w.tabs.setCurrentIndex(2); pump(8)
for name, cb in t3.sensor_checkboxes.items():
    cb.setChecked(name in keep)
pump(8)
save('tab3_iter', win(), {'sensors': rect_in(t3.sensor_scroll, w), 'plots': rect_in(t3.canvas, w)})
w.tabs.setCurrentIndex(3); pump(8); t4.refresh(); pump(4)
train_and_capture('train2')
results('tab5_iter')
json.dump({'keep': keep, 'importance': dict(zip(sens, map(float, imp)))},
          open(os.path.join(OUT, 'iteration.json'), 'w'), indent=1)

# ---------------------------------------------------------------- help, menu, reset
w.tabs.setCurrentIndex(3); pump(8)
save('help_badge', win(), {'badge': rect_in(seq_badge, w)})
base = win()
help_menu = [a for a in mb.actions() if a.menu()][0].menu()
help_menu.popup(w.mapToGlobal(QPoint(0, 0))); pump(8)
mim = img_of(help_menu); help_menu.hide()
hr = mb_rect('&Help') if '&Help' in bar else mb_rect('Help')
im, off = overlay(base, mim, pos=(hr[0], hr[1] + hr[3]), dim=False)
save('help_menu', im, {'menu': [off[0], off[1], mim.width(), mim.height()], 'help': hr})
save('menubar', win(), {'bug': mb_rect('Report a Bug'), 'star': mb_rect('★ Star on GitHub')})
reset = button(w, 'Reset')
save('reset', win(), {'reset': rect_in(reset, w)})

json.dump(SHOTS, open(os.path.join(OUT, 'shots.json'), 'w'), indent=1)
print('DONE', len(SHOTS), 'shots')
