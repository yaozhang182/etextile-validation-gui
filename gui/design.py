"""
Design tokens — the single source of truth for the application's appearance.

Both the Qt stylesheet and the matplotlib figures read their colours, fonts and
spacing from here, so the embedded plots and the surrounding widgets cannot drift
apart. Four of the five tabs are mostly plot, so that consistency is most of what
makes the app look deliberate rather than assembled.

Changing a value here changes it everywhere. Nothing outside this module should
hard-code a hex colour.
"""

# --- Surfaces ---------------------------------------------------------------
BG          = "#e9edf2"   # window background
SURFACE     = "#ffffff"   # cards, tables, plot canvases
SURFACE_ALT = "#f2f5f8"   # zebra striping, disabled fills
BORDER      = "#c5ccd6"   # hairlines
BORDER_HOVER = "#9aa5b3"

# --- Text -------------------------------------------------------------------
TEXT        = "#16202b"
TEXT_MUTED  = "#6b7683"
TEXT_FAINT  = "#9aa3ad"
TEXT_ON_ACCENT = "#ffffff"

# --- Accent + semantics -----------------------------------------------------
ACCENT      = "#0b5c8a"
ACCENT_HOVER = "#084b71"
ACCENT_PRESS = "#063a58"
ACCENT_SOFT = "#d9e8f2"   # selected rows, current-tab underline wash

OK          = "#2e7d32"
WARN        = "#ef6c00"
DANGER      = "#c62828"
DANGER_SOFT = "#fdecea"

# --- Plot palette -----------------------------------------------------------
# Ordered for categorical series. Chosen to stay distinguishable in greyscale
# print (the paper figures) and to keep a comfortable distance from DANGER,
# which is reserved for the low-confidence warning.
SERIES = [
    "#0b5c8a",  # blue
    "#e07b39",  # orange
    "#3f8f5b",  # green
    "#8a5fb0",  # purple
    "#b5893b",  # ochre
    "#4aa3b8",  # teal
    "#c05b7e",  # rose
    "#6b7683",  # grey
]

GRID = "#e8ebee"

# --- Type -------------------------------------------------------------------
# Qt resolves the first family that exists; matplotlib is given the same list.
FONT_STACK = ["Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue",
              "DejaVu Sans", "sans-serif"]
FONT_UI = ", ".join(f'"{f}"' if " " in f else f for f in FONT_STACK)
FONT_MONO = '"SF Mono", "Cascadia Mono", "DejaVu Sans Mono", monospace'

SIZE_BASE   = 14
SIZE_SMALL  = 13
SIZE_TINY   = 12
SIZE_TITLE  = 17

# --- Metrics ----------------------------------------------------------------
SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 16
RADIUS  = 4
RADIUS_SM = 3
CONTROL_H = 33
