"""
Shared chart theme for both dashboards.

One module so `dashboard.py` and `bank_dashboard.py` cannot drift into two
different-looking products, and so a contrast fix made once applies to both.

Two things here are easy to get wrong and were both wrong before:

1. **Streamlit overrides your Plotly styling.** `st.plotly_chart` applies its own
   theme by default, discarding the figure's template — which left every axis
   label at Plotly-Streamlit's #808495 (3.7:1 on white). Render through
   `show_chart()`, which passes `theme=None`.

2. **Styling belongs in a template, not in a layout dict.** Returning
   `xaxis=`/`legend=` from a layout helper collides with call sites that already
   pass those — a TypeError, not a graceful fallback. A template merges instead.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Brand ─────────────────────────────────────────────────────────────────────
NAVY   = "#1B2A4A"
GOLD   = "#C9A84C"       # accent on the navy sidebar (6.2:1) — NOT on white
WHITE  = "#FFFFFF"
GREEN  = "#276749"
RED    = "#9B2335"
BORDER = "#D1D5DB"

# ── Chart tokens ──────────────────────────────────────────────────────────────
# Brand GOLD is 2.29:1 on white — as a line or bar it is close to invisible.
# GOLD_MARK is the same hue stepped to clear the 3:1 non-text floor.
GOLD_MARK = "#B08A1E"    # 3.23:1 on white
INK       = "#1B2A4A"    # chart text        — 14.2:1 on white
INK_SOFT  = "#4A5568"    # axis ticks/titles —  7.5:1 on white
GRID      = "#E4E7EC"

# Categorical series. Validated with the dataviz skill's checker: passes the
# lightness band, chroma floor, adjacent CVD separation (worst ΔE 11.3),
# normal-vision floor (worst ΔE 17.7) and 3:1 contrast on a light surface.
# Both dashboards previously cycled near-duplicate hues (two navies, two olives,
# two greens) which is unreadable past three series. Assign in order, never cycle.
PALETTE = ["#2A6FBF", "#C2571A", "#0E8F7A", "#B08A1E", "#6B4BA8", "#C43241"]

# Single-hue sequential ramp for magnitude heatmaps. Never a multi-hue ramp:
# it has no perceptual order, goes muddy in the middle, and no single text
# colour stays legible across it.
SEQ_NAVY = ["#EDF2F8", "#D3E0EE", "#B0C7DF", "#87A8CB", "#5F86B2",
            "#41669A", "#2B4C7E", "#1B3560"]

# Diverging ramp for signed magnitudes (above/below a reference). Two hues with
# a neutral midpoint — never a hue in the middle.
DIV_REDGREEN = [[0.0, "#9B2335"], [0.5, "#F2F0EA"], [1.0, "#276749"]]

# Above this fraction of a ramp, cell labels flip to white.
TEXT_FLIP = 0.55

_AXIS = dict(
    tickfont=dict(color=INK_SOFT, size=11.5),
    title=dict(font=dict(color=INK_SOFT, size=12)),
    gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
    # theme=None hands layout back to Plotly, which honours the tight margins
    # call sites pass and then clips category names. automargin grows to fit.
    automargin=True,
)

TEMPLATE = "martindex"

pio.templates[TEMPLATE] = go.layout.Template(layout=dict(
    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    font=dict(color=INK, family="system-ui, -apple-system, sans-serif", size=12),
    xaxis=_AXIS, yaxis=_AXIS,
    legend=dict(font=dict(color=INK, size=11.5)),
    title=dict(font=dict(color=INK, size=13.5)),
    colorway=PALETTE,
))


def layout(**kw) -> dict:
    """Layout kwargs for update_layout(). Styling comes from the template."""
    base = dict(template=TEMPLATE, margin=dict(l=10, r=10, t=40, b=10))
    base.update(kw)
    return base


def show_chart(fig, **kw) -> None:
    """
    Render a figure. Always use this rather than st.plotly_chart directly —
    theme=None is what stops Streamlit discarding the template.
    """
    fig.update_layout(template=TEMPLATE)
    st.plotly_chart(fig, use_container_width=True, theme=None, **kw)


def cell_ink(value: float, vmin: float, vmax: float,
             flip: float = TEXT_FLIP) -> str:
    """
    Text colour for a heatmap cell, chosen from where the cell sits on the ramp.

    Plotly accepts only one textfont colour per trace, so a heatmap with labels
    needs per-cell annotations using this — otherwise one end of the scale is
    always unreadable.
    """
    span = (vmax - vmin) or 1
    return WHITE if (value - vmin) / span >= flip else INK


def diverging_cell_ink(value: float, vmin: float, vmax: float) -> str:
    """
    Same idea for a diverging ramp, where BOTH ends are dark and the middle is
    pale — so the flip has to happen twice.
    """
    span = (vmax - vmin) or 1
    frac = (value - vmin) / span
    return WHITE if (frac <= 0.22 or frac >= 0.78) else INK
