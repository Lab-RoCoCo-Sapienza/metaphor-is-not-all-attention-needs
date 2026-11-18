"""Render the coefficient charts used in the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# ── Constants matching the project's ``_gridlines_minor_legend12`` style ─────

TEXT_COLOR = "#1a1a1a"          # c / p / ticks / legend
BOLD_TEXT_COLOR = "#000000"     # y-axis label + feature-name band
GRID_MAJOR_COLOR = "#D8D8D8"
GRID_MINOR_COLOR = "#ECECEC"
CTX_DIVIDER_COLOR = "0.55"      # category-boundary verticals
CTX_UNDERLINE_COLOR = "0.7"     # ctx-block underlines

LABEL_SIZE = 15                 # y-axis label
TICK_LABEL_SIZE = 10
SECTION_LABEL_SIZE = 9.5        # p0/p1/p2
SECTION_LABEL_Y = -0.075        # axes-y for top of p text
CTX_LABEL_SIZE = 11             # c0..c3 (xtick labels)
CTX_TICK_PAD = 4.5
CTX_UNDERLINE_Y = -0.065
CAT_BAND_SIZE = 16              # feature-name bands
CAT_BAND_Y = 1.04
LEGEND_SIZE = 12

CATEGORIES = (
    "FIGURATIVE", "FUNCTION_WORD", "HARMFUL_PAYLOAD",
    "PUNCTUATION", "SETUP", "TECHNICAL",
)
CONTEXTS = ("c0", "c1", "c2", "c3")
SECTIONS = ("p0", "p1", "p2")


# ── Theme + feature-name parsing ────────────────────────────────────────────


def apply_paper_theme() -> None:
    """Idempotent seaborn-paper-ticks theme matching the project's charts."""
    sns.set_theme(context="paper", style="ticks", font="serif")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    })


def parse_feature_name(name: str) -> tuple[str, str, str]:
    """Split ``c{ctx}_p{section}_{CATEGORY}`` into ``(ctx, section, category)``.

    Falls back to ``("", "", name)`` for non-matching names so the chart
    still renders (the broken feature ends up at the end of the order).
    """
    parts = name.split("_", 2)
    if len(parts) < 3:
        return ("", "", name)
    return parts[0], parts[1], parts[2]


def humanize(label: str) -> str:
    """``HARMFUL_PAYLOAD`` -> ``Harmful Payload`` for the cat-band display."""
    if "_" in label or label.isupper():
        return label.replace("_", " ").title()
    return label


def _ordered_features(feature_names: list[str]) -> list[str]:
    """Sort by (category-canonical, context, section)."""
    def key(f: str) -> tuple[int, int, int]:
        ctx, sec, cat = parse_feature_name(f)
        return (
            CATEGORIES.index(cat) if cat in CATEGORIES else len(CATEGORIES),
            CONTEXTS.index(ctx) if ctx in CONTEXTS else len(CONTEXTS),
            SECTIONS.index(sec) if sec in SECTIONS else len(SECTIONS),
        )
    return sorted(feature_names, key=key)


# ── X-axis decoration: categories / ctx ticks / section labels ──────────────


def _decorate_x_axis(ax: plt.Axes, feature_order: list[str]) -> None:
    parsed = [parse_feature_name(f) for f in feature_order]
    cats_in_order = [p[2] for p in parsed]

    # Category-band labels (and the strong vertical dividers between
    # adjacent categories).
    boundaries: list[tuple[int, str]] = []
    for i, cat in enumerate(cats_in_order):
        if i == 0 or cat != cats_in_order[i - 1]:
            boundaries.append((i, cat))
    boundaries.append((len(feature_order), ""))

    for i, _ in boundaries[1:-1]:
        ax.axvline(i - 0.5, color=CTX_DIVIDER_COLOR, lw=0.8, alpha=0.7)

    for k in range(len(boundaries) - 1):
        i, cat = boundaries[k]
        j = boundaries[k + 1][0]
        ax.text(
            (i + j - 1) / 2, CAT_BAND_Y, humanize(cat),
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=CAT_BAND_SIZE, color=BOLD_TEXT_COLOR,
        )

    # Ctx ticks (c0..c3) — one tick per (cat, ctx) block, plus a faint
    # horizontal underline spanning each block.
    ctx_block_positions: dict[tuple[str, str], list[int]] = {}
    for i, (ctx, _sec, cat) in enumerate(parsed):
        ctx_block_positions.setdefault((cat, ctx), []).append(i)

    ctx_positions: list[float] = []
    ctx_labels: list[str] = []
    for (_cat, ctx), idxs in ctx_block_positions.items():
        ctx_positions.append(float(np.mean(idxs)))
        ctx_labels.append(ctx)
        ax.plot(
            [min(idxs) - 0.35, max(idxs) + 0.35],
            [CTX_UNDERLINE_Y, CTX_UNDERLINE_Y],
            transform=ax.get_xaxis_transform(),
            color=CTX_UNDERLINE_COLOR, lw=0.4, clip_on=False,
        )

    ax.set_xticks(ctx_positions)
    ax.set_xticklabels(ctx_labels, fontsize=CTX_LABEL_SIZE)
    ax.tick_params(axis="x", length=0)

    # Section labels (p0..p2) under each individual bar.
    for i, (_ctx, sec, _cat) in enumerate(parsed):
        ax.text(
            i, SECTION_LABEL_Y, sec,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=SECTION_LABEL_SIZE, color="0.4",
        )


# ── Main render ─────────────────────────────────────────────────────────────


def render_chart(
    beta_df: pd.DataFrame,
    out_path: Path,
    *,
    up_class: str,
    down_class: str,
    positive_class: str | None = None,
    red_class: str | None = None,
    y_step: float = 0.5,
    show_errorbars: bool = True,
) -> None:
    """Render one publication chart from a β CSV.

    Args:
        beta_df: DataFrame with ``feature_name``, ``beta_mean`` (and,
            when ``show_errorbars=True``, ``beta_std``).
        out_path: Output PNG path. Parent dirs created as needed.
        up_class: Name of the class whose features point upward (positive bars).
        down_class: Name of the class whose features point downward.
        positive_class: The class for which the *underlying* logistic
            regression's β is positive. Defaults to ``up_class`` (the
            common case where the chart agrees with the model's class
            ordering); pass the model's positive class if you want the
            chart to invert it (e.g. show "predicts safe" upward when
            the model treats "unsafe" as positive).
        red_class: When set, ensures bars labelled "predicts <red_class>"
            are rendered red regardless of whether they go up or down.
        y_step: Major y-axis tick step. The minor grid is drawn at half
            this value.
        show_errorbars: Whether to draw ±1σ error bars on each bar.
    """
    apply_paper_theme()

    # Pre-process: order features and decide sign + cmap.
    feature_order = _ordered_features(beta_df["feature_name"].tolist())
    df = beta_df.set_index("feature_name").reindex(feature_order)

    positive_class = positive_class or up_class
    sign = 1.0 if up_class == positive_class else -1.0
    display_values = sign * df["beta_mean"].to_numpy()
    stds = df["beta_std"].to_numpy() if show_errorbars and "beta_std" in df.columns else None

    cmap_name = (
        "RdBu_r" if red_class is not None and up_class == red_class else "RdBu"
    )
    cmap = plt.get_cmap(cmap_name)
    vmax = float(np.abs(display_values).max() or 1e-9)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    bar_colors = [cmap(norm(v)) for v in display_values]

    # Draw bars.
    n = len(feature_order)
    centers = np.arange(n)
    fig, ax = plt.subplots(figsize=(14.0, 4.6), constrained_layout=True)
    ax.bar(
        centers, display_values, width=0.8, yerr=stds, color=bar_colors,
        edgecolor="0.4", linewidth=0.4,
        error_kw={"ecolor": "0.35", "elinewidth": 0.6, "capsize": 1.5},
    )
    if (display_values < 0).any():
        ax.axhline(0, color="0.3", lw=0.6, zorder=2)

    # Tight horizontal margins (default xmargin 0.05 wastes ~5
    # bar-widths on each side).
    ax.margins(x=0.005)

    # 3-tier x-axis (cat band / ctx ticks / section labels).
    _decorate_x_axis(ax, feature_order)
    ax.set_xlabel("")
    ax.set_ylabel(
        r"Feature contribution ($\beta$)",
        fontsize=LABEL_SIZE, color=BOLD_TEXT_COLOR,
    )

    # Horizontal gridlines (major + minor) behind the bars.
    ax.set_axisbelow(True)
    ax.yaxis.grid(
        True, color=GRID_MAJOR_COLOR, linewidth=0.7, linestyle="-",
        which="major", zorder=0,
    )
    ax.xaxis.grid(False)
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.yaxis.set_minor_locator(MultipleLocator(y_step / 2.0))
    ax.yaxis.grid(
        True, color=GRID_MINOR_COLOR, linewidth=0.5, linestyle="-",
        which="minor", zorder=0,
    )
    ax.tick_params(
        axis="y", which="minor", length=2.5, width=0.5, color=TEXT_COLOR,
    )
    ax.tick_params(
        axis="x", labelsize=CTX_LABEL_SIZE, labelcolor=TEXT_COLOR,
        pad=CTX_TICK_PAD,
    )
    ax.tick_params(
        axis="y", labelsize=TICK_LABEL_SIZE, labelcolor=TEXT_COLOR,
        pad=CTX_TICK_PAD,
    )

    # Headroom so the upper-right legend doesn't sit on the tallest bars.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(top=ymax + (ymax - ymin) * 0.18)

    # Custom swatch legend — vertical layout, with frame, extra padding.
    legend_handles = [
        Patch(
            label=f"predicts {up_class}",
            facecolor=cmap(0.85), edgecolor="0.4", linewidth=0.4,
        ),
        Patch(
            label=f"predicts {down_class}",
            facecolor=cmap(0.15), edgecolor="0.4", linewidth=0.4,
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", ncol=1,
        fontsize=LEGEND_SIZE, frameon=True,
        borderpad=0.7, labelspacing=0.7,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
