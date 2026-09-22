"""
Contextual help badges.

A small "?" next to a control. Hovering shows a few sentences explaining what
the control is for; clicking opens the manual at the matching section.

Qt tooltips render rich text but their links are not clickable, so the two roles
are split: the tooltip carries the explanation, the click carries the navigation.
Every badge says so, otherwise nobody would discover the click.

Each entry lives in HELP below rather than being written inline at the call
site, so the wording can be reviewed in one place and kept consistent with
docs/manual.md.
"""

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from gui import design, icons

#: key -> (title, body_html, manual_anchor)
#:
#: Bodies are short: two to four sentences. Anything longer belongs in the
#: manual, which is one click away.
HELP = {
    # ---------------------------------------------------------- Tab 1
    'train_subjects': (
        "Training subjects",
        "One <b>subject</b> is one recording session, and it needs three files "
        "that belong together: the video, the sensor CSV, and the "
        "video-timestamp CSV. All three must share the same clock "
        "(Unix seconds) — that is how a sensor reading is matched to a video "
        "frame.<br><br>The model learns from these sessions. Add as many as you "
        "have; they must all expose the same sensor channel names.",
        "#tab-1--data-import",
    ),
    'test_subjects': (
        "Test subjects",
        "Sessions held back from training, used only to score the model.<br><br>"
        "Keep at least one. A model measured on the same data it learned from "
        "always looks better than it is, so the accuracy on this held-out set is "
        "the number that actually means something.",
        "#choosing-the-traintest-split",
    ),
    'extract': (
        "Extract skeleton and compute angles",
        "Runs pose tracking on every session not yet processed, then derives the "
        "26 joint-angle channels from it. This is the slow step — roughly "
        "0.1 s per video frame, so a two-minute recording takes a few "
        "minutes.<br><br>Results are cached next to each video, so you pay this "
        "once per recording, not once per experiment.",
        "#tab-1--data-import",
    ),
    'subject_files': (
        "The three files",
        "<b>Video</b> — the webcam recording (H.264 MP4 preferred).<br>"
        "<b>Sensor CSV</b> — a timestamp column plus one numeric column per "
        "sensing element.<br>"
        "<b>Timestamps CSV</b> — one row per video frame, with "
        "<tt>FrameIndex</tt> and <tt>EpochTime</tt>.<br><br>"
        "All three must use the same absolute clock. The files are checked "
        "before they are accepted, and any problem is explained here.",
        "#input-requirements",
    ),
    # ---------------------------------------------------------- Tab 2
    'joint_colours': (
        "Why the joints are coloured",
        "Each joint is coloured by the pose tracker's own confidence for that "
        "frame — green is well tracked, red is not, usually because the limb was "
        "occluded.<br><br>Scrub the timeline and watch for limbs turning red: "
        "that is where the ground truth is weakest, so any accuracy figure for "
        "those channels means less.",
        "#tab-2--skeleton-viewer",
    ),
    # ---------------------------------------------------------- Tab 3
    'angle_targets': (
        "Joint angles (targets)",
        "What you tick here is what the model will try to predict from the "
        "sensors. Start with two or three related channels rather than all 26 — "
        "a focused model trains faster and is far easier to interpret.<br><br>"
        "Pick angles the garment could plausibly sense: a sleeve cannot measure "
        "knee flexion.<br><br><b>Click for what each channel means</b>, "
        "including which direction counts as positive.",
        "#joint-angle-definitions",
    ),
    'angle_confidence': (
        "The number in brackets",
        "Mean tracking confidence for that channel, from 0 to 1, over the "
        "selected session. Channels shown in red fell below the threshold at "
        "which the pose tracker considers a landmark occluded.<br><br>"
        "This matters when choosing targets: a channel whose <i>ground truth</i> "
        "was poorly tracked will score badly however good the garment is.",
        "#tab-3--angles--sensors",
    ),
    'sensor_inputs': (
        "Sensor channels (inputs)",
        "The garment channels the model is allowed to use. Leave them all ticked "
        "unless you are deliberately testing a subset.<br><br>Only channels "
        "present in <i>every</i> session appear here — if one session has extra "
        "channels, a note says which are being ignored.",
        "#tab-3--angles--sensors",
    ),
    # ---------------------------------------------------------- Tab 4
    'training_selection': (
        "Where these come from",
        "This line summarises what will be trained, and is <b>not editable "
        "here</b>.<br><br>Subjects come from <b>Tab 1</b> — add or remove them "
        "there. Targets and inputs come from the checkboxes in <b>Tab 3</b>.<br>"
        "<br>If it says 0 subjects, no session finished extraction yet.",
        "#changing-targets-and-inputs",
    ),
    'sequence_length': (
        "Sequence length",
        "How many consecutive frames of sensor history the model sees for each "
        "prediction. 40 frames is about 1.3 s at 30 fps.<br><br>"
        "<b>This one bites:</b> a session with fewer aligned samples than the "
        "sequence length contributes nothing and is skipped. If the log reports "
        "skipped sessions, lower this.",
        "#hyperparameters",
    ),
    'time_alignment': (
        "Time alignment",
        "The sensor and the video run at different rates, so one has to be "
        "resampled onto the other's clock.<br><br><b>Align to video frames</b> "
        "(the default) interpolates the sensor onto each video frame, which is "
        "where the ground truth is defined and gives the most samples. The "
        "alternative aligns to sensor samples instead, yielding fewer points.",
        "#hyperparameters",
    ),
    # ---------------------------------------------------------- Tab 5
    'headline_metrics': (
        "The four metrics",
        "<b>MPJAE</b> — mean absolute error in degrees. Lower is better.<br>"
        "<b>AMPE</b> — the same error as a percentage of the angle's range.<br>"
        "<b>RMSE</b> — error in degrees, punishing large mistakes more.<br>"
        "<b>PCC</b> — correlation with the truth, −1 to 1; closer to 1 is "
        "better.<br><br>High PCC with high MPJAE means the sensor follows the "
        "<i>shape</i> of the movement but is offset or scaled — often fixable by "
        "calibration.",
        "#reading-the-numbers",
    ),
    'per_subject': (
        "Per-subject results",
        "The same metrics for each test session on its own.<br><br>Read this "
        "before the pooled numbers: one unusual wearer can carry the average, "
        "and a garment that works for three people but not a fourth is a "
        "different finding from one that works moderately for everyone.",
        "#tab-5--results",
    ),
    'tracking_conf_column': (
        "Tracking conf. column",
        "How well the <i>ground truth</i> for each channel was tracked — not how "
        "good the sensor is.<br><br>It reframes every error next to it. A large "
        "error on a low-confidence channel says little about the garment, because "
        "the reference itself was unreliable. A large error on a well-tracked "
        "channel is a real result.",
        "#reading-the-numbers",
    ),
    'prediction_curves': (
        "Prediction vs ground truth",
        "One panel per predicted channel, over the whole test recording. Black "
        "is the ground truth from the video, red dashed is what the model "
        "inferred from the sensors alone.<br><br>This is the figure to judge "
        "the garment by: the tables give one number per channel, but only the "
        "curves show <i>where</i> the error is — a constant offset, drift over "
        "time, or a specific movement the sensors miss entirely.",
        "#tab-5--results",
    ),
    'sensor_importance': (
        "Sensor importance",
        "How much the trained model relied on each channel, from the first "
        "convolution layer's weights.<br><br>This is the part that feeds back "
        "into garment design: a channel with near-zero importance is not "
        "contributing to this movement and could be moved, merged, or dropped in "
        "the next revision.",
        "#tab-5--results",
    ),
}


class HelpBadge(QPushButton):
    """A "?" that explains on hover and opens the manual on click."""

    def __init__(self, key, parent=None):
        super().__init__(parent)
        if key not in HELP:
            raise KeyError(f"no help text registered for {key!r}")
        title, body, anchor = HELP[key]
        self._anchor = anchor

        self.setProperty("subtle", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(QSize(20, 20))
        self.setFlat(True)
        self.setAutoDefault(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        badge = icons.icon('circle-help', design.TEXT_MUTED, 15)
        if badge.isNull():
            self.setText("?")          # fall back to a literal glyph
        else:
            self.setIcon(badge)
            self.setIconSize(QSize(15, 15))

        self.setToolTip(
            f"<div style='max-width:340px'>"
            f"<b>{title}</b><br><br>{body}"
            f"<br><br><i style='color:{design.TEXT_FAINT}'>"
            f"Click to open the manual.</i></div>"
        )
        self.setAccessibleName(f"Help: {title}")
        self.clicked.connect(self._open)

    def _open(self):
        from gui.main_window import open_manual
        open_manual(self._anchor)


def labelled(text, key, bold=False):
    """
    A label with a help badge after it, as one widget.

    Use where a QLabel would go. Group-box titles cannot host a child widget,
    so those get a header row built with this instead.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    label = QLabel(f"<b>{text}</b>" if bold else text)
    row.addWidget(label)
    row.addWidget(HelpBadge(key))
    row.addStretch()
    return holder


def attach(layout, key):
    """Append a badge to an existing row layout."""
    badge = HelpBadge(key)
    layout.addWidget(badge)
    return badge
