#!/usr/bin/env python3
"""
Tests for input validation.

These cover the mistakes a non-technical user actually makes when picking three
files out of a dialog. Before validation existed, most of them surfaced either as
a raw pandas exception or minutes into a pose-extraction run — the swapped-CSV
case produced no complaint at all, because both files parse as valid CSVs.

Run:  PYTHONPATH=. python tests/test_validation.py
"""

import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from core.validation import (  # noqa: E402
    ERROR, INFO, WARNING, Finding, format_findings, has_errors, validate_session,
)

EXAMPLES = Path(__file__).resolve().parent.parent / 'input_data' / 'examples_data'
S1 = EXAMPLES / 'session_1'
S4 = EXAMPLES / 'session_4'

_tmp = None


def tmpdir():
    global _tmp
    if _tmp is None:
        _tmp = Path(tempfile.mkdtemp())
    return _tmp


def titles(findings):
    return [f.title for f in findings]


def _skip_without_examples():
    if not (S1 / 'video.mp4').exists():
        raise RuntimeError(f"example data missing at {S1}")


def test_valid_session_is_accepted():
    _skip_without_examples()
    findings = validate_session(S1 / 'video.mp4', S1 / 'sensor.csv', S1 / 'frames.csv')
    assert not has_errors(findings), titles(findings)
    assert not [f for f in findings if f.level == WARNING], titles(findings)


def test_unreadable_video_is_blocked():
    bad = tmpdir() / 'notavideo.mp4'
    bad.write_bytes(b'this is not a video')
    findings = validate_session(bad, S1 / 'sensor.csv', S1 / 'frames.csv')
    assert has_errors(findings)
    assert any('Video cannot be read' == f.title for f in findings), titles(findings)


def test_swapped_csvs_are_detected():
    """Both files parse, so nothing else would catch this."""
    _skip_without_examples()
    findings = validate_session(S1 / 'video.mp4', S1 / 'frames.csv', S1 / 'sensor.csv')
    assert has_errors(findings)
    assert any('swapped' in f.title.lower() for f in findings), titles(findings)


def test_non_overlapping_streams_are_blocked():
    """A sensor CSV from a different recording session."""
    _skip_without_examples()
    findings = validate_session(S1 / 'video.mp4', S4 / 'sensor.csv', S1 / 'frames.csv')
    assert has_errors(findings)
    hit = [f for f in findings if 'do not overlap' in f.title]
    assert hit, titles(findings)
    assert 'same time as this video' in hit[0].fix


def test_sensor_without_timestamp_column_is_blocked():
    _skip_without_examples()
    df = pd.read_csv(S1 / 'sensor.csv', sep=';').drop(columns=['EpochTime'])
    path = tmpdir() / 'no_time.csv'
    df.to_csv(path, sep=';', index=False)
    findings = validate_session(S1 / 'video.mp4', path, S1 / 'frames.csv')
    assert has_errors(findings)
    assert any('no timestamp column' in f.title for f in findings), titles(findings)


def test_sensor_without_channels_is_blocked():
    _skip_without_examples()
    df = pd.read_csv(S1 / 'sensor.csv', sep=';')[['EpochTime']]
    path = tmpdir() / 'no_channels.csv'
    df.to_csv(path, sep=';', index=False)
    findings = validate_session(S1 / 'video.mp4', path, S1 / 'frames.csv')
    assert has_errors(findings)
    assert any('no sensor channels' in f.title for f in findings), titles(findings)


def test_relative_timestamps_are_blocked():
    """Time measured from the start of the recording cannot be aligned."""
    _skip_without_examples()
    df = pd.read_csv(S1 / 'sensor.csv', sep=';')
    df['EpochTime'] -= df['EpochTime'].iloc[0]
    path = tmpdir() / 'relative.csv'
    df.to_csv(path, sep=';', index=False)
    findings = validate_session(S1 / 'video.mp4', path, S1 / 'frames.csv')
    assert has_errors(findings)
    assert any('wall-clock' in f.title for f in findings), titles(findings)


def test_frames_without_epochtime_is_blocked():
    _skip_without_examples()
    df = pd.read_csv(S1 / 'frames.csv', sep=';').rename(columns={'EpochTime': 't'})
    path = tmpdir() / 'bad_frames.csv'
    df.to_csv(path, sep=';', index=False)
    findings = validate_session(S1 / 'video.mp4', S1 / 'sensor.csv', path)
    assert has_errors(findings)
    hit = [f for f in findings if 'no EpochTime' in f.title]
    assert hit, titles(findings)
    # The columns that *were* found must be listed, so the user can see what
    # the file actually contains. 't' is not close enough to any known
    # timestamp name to be suggested, which is correct.
    assert 'FrameIndex' in hit[0].detail and 'Marker' in hit[0].detail

    # A near-miss name, however, should be pointed at.
    near = pd.read_csv(S1 / 'frames.csv', sep=';').rename(
        columns={'EpochTime': 'epoch_time'})
    path2 = tmpdir() / 'near_frames.csv'
    near.to_csv(path2, sep=';', index=False)
    hit2 = [f for f in validate_session(S1 / 'video.mp4', S1 / 'sensor.csv', path2)
            if 'no EpochTime' in f.title]
    assert hit2 and 'epoch_time' in hit2[0].detail, "near-miss should be suggested"


def test_missing_and_empty_files_are_blocked():
    _skip_without_examples()
    empty = tmpdir() / 'empty.csv'
    empty.touch()
    assert has_errors(validate_session(S1 / 'video.mp4', empty, S1 / 'frames.csv'))

    absent = tmpdir() / 'does_not_exist.csv'
    findings = validate_session(S1 / 'video.mp4', absent, S1 / 'frames.csv')
    assert has_errors(findings)
    # user-facing text must not leak a Python repr
    assert 'PosixPath' not in format_findings(findings, html=False)


def test_comma_separated_csv_is_accepted():
    _skip_without_examples()
    df = pd.read_csv(S1 / 'sensor.csv', sep=';')
    path = tmpdir() / 'comma.csv'
    df.to_csv(path, index=False)          # ',' separated
    assert not has_errors(validate_session(S1 / 'video.mp4', path, S1 / 'frames.csv'))


def test_wrong_container_fps_is_reported_as_handled():
    """The example MP4s claim 25 fps but were captured at ~30."""
    _skip_without_examples()
    findings = validate_session(S1 / 'video.mp4', S1 / 'sensor.csv', S1 / 'frames.csv')
    hit = [f for f in findings if 'frame-rate tag' in f.title]
    assert hit, titles(findings)
    assert hit[0].level == INFO, "a handled discrepancy must not alarm the user"


def test_every_finding_is_actionable():
    """A problem the user cannot act on is a bug report, not a message."""
    cases = [
        (tmpdir() / 'x.mp4', S1 / 'sensor.csv', S1 / 'frames.csv'),
        (S1 / 'video.mp4', S1 / 'frames.csv', S1 / 'sensor.csv'),
        (S1 / 'video.mp4', S4 / 'sensor.csv', S1 / 'frames.csv'),
    ]
    for args in cases:
        for f in validate_session(*args):
            if f.level in (ERROR, WARNING):
                assert f.fix, f"{f.title} has no suggested fix"
                assert f.detail, f"{f.title} has no detail"


def test_validation_never_raises():
    """Garbage in must produce findings, not a traceback."""
    junk = tmpdir() / 'junk.bin'
    junk.write_bytes(os.urandom(2048))
    for args in [(junk, junk, junk), ('', '', ''), (None, None, None)]:
        findings = validate_session(*args)
        assert has_errors(findings), args


def test_html_and_text_rendering():
    findings = [Finding(ERROR, 'T', 'D', 'F')]
    html = format_findings(findings)
    assert 'T' in html and 'D' in html and 'F' in html and '<b' in html
    text = format_findings(findings, html=False)
    assert text.startswith('[ERROR]') and '<' not in text
    assert format_findings([]) == ''


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
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
