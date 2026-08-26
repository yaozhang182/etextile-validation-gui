"""
Shared application state helpers.

The GUI keeps one entry per recording session ("subject") in
state['train_subjects'] / state['test_subjects']. A subject bundles the three
files the user uploads together (video, sensor CSV, video-timestamp CSV) with
everything derived from them.

Lives in its own module so panels can import these helpers without importing
main_window (which imports the panels).
"""


SPLITS = ('train', 'test')


def make_subject(subject_id, video_path, sensor_path, video_csv_path):
    """Create a fresh subject record from the three uploaded files."""
    return {
        'id': subject_id,
        'video_path': video_path,
        'sensor_path': sensor_path,
        'video_csv_path': video_csv_path,
        # Filled in when the files are parsed (Tab 1)
        'sensor_df': None,
        'video_df': None,
        # Filled in by skeleton extraction (Tab 1)
        'skeleton': None,   # dict from extract_skeleton_from_video
        'angles': None,     # {angle_name: (T,) array}
        'time': None,       # (T,) seconds, relative to video start
        'status': 'pending',  # pending | extracting | ready | error
        'error': None,
    }


def get_subjects(state, split):
    """All subjects for 'train' or 'test'."""
    return state.get(f'{split}_subjects', [])


def ready_subjects(state, split):
    """Subjects whose skeleton extraction finished and that have sensor data."""
    return [
        s for s in get_subjects(state, split)
        if s.get('status') == 'ready'
        and s.get('angles') is not None
        and s.get('sensor_df') is not None
    ]


def next_subject_id(state, split):
    """Auto-name the next subject S1, S2, ... avoiding collisions."""
    existing = {s['id'] for s in get_subjects(state, split)}
    i = 1
    while f'S{i}' in existing:
        i += 1
    return f'S{i}'


def sensor_columns(subject):
    """Sensor channel names for a subject, excluding timestamp/index columns."""
    df = subject.get('sensor_df')
    if df is None:
        return []
    exclude = {'epochtime', 'epoctime', 'epoch', 'time', 'timestamp',
               'frameindex', 'marker'}
    return [c for c in df.columns if c.lower() not in exclude]


def common_sensor_columns(state):
    """
    Sensor channels present in every ready subject across both splits.

    Tab 3 offers only these, so a selection can never reference a channel that
    some subject is missing.
    """
    subjects = [s for split in SPLITS for s in ready_subjects(state, split)]
    if not subjects:
        return []

    common = None
    for s in subjects:
        cols = sensor_columns(s)
        common = cols if common is None else [c for c in common if c in cols]
    return common or []


def inconsistent_sensor_subjects(state):
    """
    Subjects whose sensor channels differ from the common set.

    Returns a list of (split, subject_id, extra_columns) for warning the user.
    """
    common = set(common_sensor_columns(state))
    out = []
    for split in SPLITS:
        for s in ready_subjects(state, split):
            extra = [c for c in sensor_columns(s) if c not in common]
            if extra:
                out.append((split, s['id'], extra))
    return out
