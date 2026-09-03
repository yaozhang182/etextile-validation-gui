"""
Shared matplotlib styling for the embedded figures.

Four of the five tabs are dominated by plots, so stock matplotlib styling —
heavy black spines, grey outer frame, default blue — is what makes the whole
application look dated regardless of how the widgets are styled. Calling
apply_plot_style() once at startup fixes all of them at the source.

Colours come from gui.design so the figures and the surrounding Qt chrome stay
in step.
"""

from cycler import cycler

from gui import design

_applied = False


def apply_plot_style():
    """Install the application's matplotlib defaults. Safe to call repeatedly."""
    global _applied
    if _applied:
        return

    import matplotlib as mpl

    mpl.rcParams.update({
        # --- canvas: match the surrounding widget surface exactly, so an
        # embedded figure reads as part of the panel rather than a pasted image
        'figure.facecolor':  design.SURFACE,
        'figure.edgecolor':  design.SURFACE,
        'axes.facecolor':    design.SURFACE,
        'savefig.facecolor': design.SURFACE,
        'savefig.edgecolor': design.SURFACE,
        'figure.autolayout': False,     # panels call tight_layout themselves

        # --- type
        'font.family':     'sans-serif',
        'font.sans-serif': design.FONT_STACK,
        'font.size':        design.SIZE_TINY,
        'axes.titlesize':   design.SIZE_SMALL,
        'axes.labelsize':   design.SIZE_TINY,
        'xtick.labelsize':  design.SIZE_TINY - 1,
        'ytick.labelsize':  design.SIZE_TINY - 1,
        'legend.fontsize':  design.SIZE_TINY - 1,
        'axes.titleweight': 'medium',
        'axes.titlepad':    8,

        # --- frame: keep only the axes the reader needs
        'axes.edgecolor':   design.BORDER,
        'axes.linewidth':   0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.labelcolor':  design.TEXT_MUTED,
        'text.color':       design.TEXT,
        'xtick.color':      design.TEXT_FAINT,
        'ytick.color':      design.TEXT_FAINT,
        'xtick.labelcolor': design.TEXT_MUTED,
        'ytick.labelcolor': design.TEXT_MUTED,
        'xtick.direction':  'out',
        'ytick.direction':  'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,

        # --- grid: present but never competing with the data
        'axes.grid':       True,
        'axes.grid.axis':  'both',
        'grid.color':      design.GRID,
        'grid.linewidth':  0.8,
        'grid.alpha':      1.0,
        'axes.axisbelow':  True,

        # --- data
        'axes.prop_cycle': cycler(color=design.SERIES),
        'lines.linewidth': 1.4,
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': design.SURFACE,
        'patch.linewidth': 0.6,

        # --- legend: a light card rather than a boxed frame
        'legend.frameon':     True,
        'legend.facecolor':   design.SURFACE,
        'legend.edgecolor':   design.BORDER,
        'legend.framealpha':  0.92,
        'legend.borderpad':   0.5,
        'legend.labelspacing': 0.3,
        'legend.handlelength': 1.6,
    })
    _applied = True


def style_3d_axes(ax):
    """
    Tone down a 3D axes.

    mplot3d ignores most rcParams: it paints opaque grey panes and dark grid
    lines that clash badly with a light theme. This has to be applied per-axes.
    """
    try:
        for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane_axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            pane_axis.pane.set_edgecolor(design.BORDER)
            pane_axis.pane.set_alpha(1.0)
            pane_axis._axinfo['grid'].update({
                'color': design.GRID, 'linewidth': 0.7,
            })
    except Exception:
        pass    # mplot3d internals differ between versions; styling is optional
