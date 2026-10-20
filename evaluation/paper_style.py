"""
plot_style.py

Shared styling utility for CFE-Net (JSTARS) paper figures.
Import and call `apply_style()` once at the top of any plotting script.

Design decisions baked in (per paper discussion):
- No in-plot titles (LaTeX captions handle this)
- Independent y-axis scales across small-multiple panels (raw counts, not normalized)
- Serif font (Liberation Serif -- metrically compatible with Times New Roman)
- Vector-first: always save as PDF/EPS, not PNG, unless explicitly rasterizing a
  complex/high-cell-count figure (e.g. dense heatmaps), in which case use 300+ DPI.
- One consistent qualitative palette for the six model configs, reused across
  every figure type (bars, lines, heatmap category axes, etc.)
- One diverging colormap for delta/heatmap figures (e.g. CFE on/off deltas),
  centered at zero.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# Six model configs: {UNet, UPerNet, DeepLabV3+} x {CFE off, CFE on}
# Paired so CFE-off / CFE-on share a hue family but differ in saturation/value --
# this lets a reader visually group by architecture first, CFE second.
CONFIG_COLORS = {
    "unet_off":      "#7FA8C9",  # muted blue
    "unet_on":       "#1B4F72",  # deep blue
    "upernet_off":   "#E8B074",  # muted orange
    "upernet_on":    "#B5651D",  # deep orange
    "deeplab_off":   "#9FC98A",  # muted green
    "deeplab_on":    "#256E32",  # deep green
}

# Flat ordered list, for cases needing a simple qualitative sequence
CONFIG_COLOR_LIST = list(CONFIG_COLORS.values())

# Single-series default (e.g. one dataset's histogram bars)
DEFAULT_BAR_COLOR = "#3B6E9E"
DEFAULT_BAR_EDGE = "#1C2B36"

# Diverging colormap for delta/heatmap figures (CFE helps vs hurts), centered at 0
DIVERGING_CMAP = "RdBu_r"

# Dataset colors (WHU, Massachusetts, INRIA) -- used for small-multiple histograms
# and any figure comparing the three source datasets directly
DATASET_COLORS = {
    "WHU": "#3B6E9E",
    "Massachusetts": "#B5651D",
    "INRIA": "#256E32",
}

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def apply_style(font_family="serif", base_font="Liberation Serif", font_size=10):
    """
    Apply global matplotlib rcParams for paper-ready figures.

    Call this once per script, before creating any figures.

    Parameters
    ----------
    font_family : str
        'serif' matches IEEE/JSTARS body text (Times-family). Use 'sans-serif'
        only if you have a specific reason to deviate.
    base_font : str
        Concrete font to request. 'Liberation Serif' is metrically compatible
        with Times New Roman and is available in this environment. If you have
        actual Times New Roman installed locally, swap this to 'Times New Roman'.
    font_size : int
        Base font size in points. 9-10pt reads well at typical two-column
        IEEE figure widths (~3.5in single column, ~7.16in double column).
    """
    mpl.rcParams.update({
        # Fonts
        "font.family": font_family,
        "font.serif": [base_font, "Times New Roman", "DejaVu Serif"],
        "font.size": font_size,
        "axes.titlesize": font_size,       # unused (no in-plot titles) but set for safety
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size,

        # Math text matches serif body font
        "mathtext.fontset": "stix",

        # Remove chartjunk
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.color": "#888888",
        "axes.axisbelow": True,

        # No in-plot titles by convention (captions handle this) --
        # this doesn't suppress titles programmatically, just a reminder:
        # don't call ax.set_title() in figure-generation scripts.

        # Line/marker defaults
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "patch.linewidth": 0.6,

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.5,

        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed fonts as TrueType, not Type 3 -- avoids
        "ps.fonttype": 42,    # font-substitution issues in IEEE's PDF pipeline
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_axis(ax, hide_top_right=True):
    """Apply per-axis cleanup that rcParams can't fully cover."""
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.8)


def noise_floor_line(ax, x=2.0, label=None, color="#C0392B"):
    """
    Draw the noise-floor cutoff line (2 m^2 default) in a consistent style
    across every figure that needs it (dataset histograms, noise-floor
    justification figure, size-bin figures).
    """
    ax.axvline(x=x, color=color, linestyle="--", linewidth=1.3, zorder=3,
               label=label)


def size_bin_lines(ax, edges=(2, 75, 300, 1000, 5000), color="#555555"):
    """
    Draw vertical lines at size-bin boundaries. Use on instance-area
    histograms / size-bin justification figures for visual consistency
    with the noise floor line above.
    """
    for e in edges:
        ax.axvline(x=e, color=color, linestyle=":", linewidth=0.9,
                   alpha=0.7, zorder=2)


def logspace_bins(data_min, data_max, n_bins=40):
    """Generate log-spaced bin edges for instance-area histograms."""
    return np.logspace(np.log10(data_min), np.log10(data_max), n_bins)


def save_fig(fig, path, vector=True):
    """
    Save a figure per paper conventions: vector (PDF) by default.
    Pass vector=False only for genuinely complex/high-cell-count rasters
    (e.g. dense heatmaps), which will still be saved at 300 DPI.
    """
    if vector and not str(path).endswith((".pdf", ".eps", ".svg")):
        path = str(path) + ".pdf"
    fig.savefig(path)
    return path