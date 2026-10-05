"""plot_measurements.py | Read the measurements and draw them.

This file only reads what measurements.py wrote. It never renders or measures audio,
so plots can be redone as often as you like without touching the GPU.

Command line
------------
    # plots for an evolution run (reads the .jsonl logbook)
    python plot_measurements.py run logbook.jsonl --out plots/

    # voice map for a folder that was measured with `measurements.py measure`
    python plot_measurements.py voices voices.csv --out plots/

    # no data yet? Make a simulated logbook and plot it, to see what you will get
    python plot_measurements.py demo --out plots_demo/

Plots made by `run`
-------------------
    fitness.png        best fitness per generation, one line per character
    fitness_terms.png  validity, Wundt novelty and distinctiveness of the best individual
    behaviour_map.png  2-D map of behaviour space: where each character ended up
    profiles.png       what each character sounds like, feature by feature
    separation.png     how far apart the characters are, per generation

Dependencies: numpy, pandas, matplotlib
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

FEATURE_NAMES = ["tempo", "pause", "pitch_level", "pitch_range", "loud_dyn", "brightness"]
FEATURE_LABELS = {
    "tempo": "Tempo",
    "pause": "Pause share",
    "pitch_level": "Pitch level",
    "pitch_range": "Pitch range",
    "loud_dyn": "Loudness dynamics",
    "brightness": "Brightness",
}
FEATURE_ENDS = {  # (what a negative value sounds like, what a positive value sounds like)
    "tempo": ("slower", "faster"),
    "pause": ("fluent", "hesitant"),
    "pitch_level": ("lower", "higher"),
    "pitch_range": ("flat", "lively"),
    "loud_dyn": ("even", "dramatic"),
    "brightness": ("husky", "clear"),
}
B_COLS = [f"b_{n}" for n in FEATURE_NAMES]

# --------------------------------------------------------------------------- #
# Style: one place for every colour and size
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Fixed order. The first three stay distinguishable for colour-blind readers in a
# scatter plot, so keep to three characters if you can. Never reorder by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # shape repeats the colour, for print

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "axes.labelcolor": INK_2, "text.color": INK,
    "lines.linewidth": 2.0, "lines.markersize": 7,
    "legend.frameon": False,
})


def _colours(characters: list[str]) -> dict[str, str]:
    """Colour follows the character name (sorted), not its rank in any one plot."""
    if len(characters) > len(SERIES):
        raise SystemExit(f"{len(characters)} characters is more than {len(SERIES)} colours. Plot fewer.")
    return {c: SERIES[i] for i, c in enumerate(sorted(characters))}


def _markers(characters: list[str]) -> dict[str, str]:
    return {c: MARKERS[i] for i, c in enumerate(sorted(characters))}


def _title(fig, title: str, subtitle: str, note: str = "") -> None:
    fig.text(0.06, 0.965, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.06, 0.915, subtitle, fontsize=10, color=INK_2, ha="left", va="top")
    if note:
        fig.text(0.98, 0.015, note, fontsize=9, fontweight="bold", color=INK_2, ha="right", va="bottom")


def _end_labels(ax, ends: list[tuple[float, float, str]], min_gap_frac: float = 0.06) -> None:
    """Name each line at its right end. Labels are nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    placed: list[float] = []
    for x, y, name in sorted(ends, key=lambda e: e[1]):
        y_text = y if not placed else max(y, placed[-1] + gap)
        placed.append(y_text)
        ax.annotate(name, (x, y_text), xytext=(10, 0), textcoords="offset points",
                    va="center", ha="left", color=INK, fontsize=10, annotation_clip=False)


def _label_points(fig, ax, xs, ys, names) -> None:
    """Name every point, trying a few positions so labels do not sit on each other."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tries = [(9, 7, "left"), (9, -14, "left"), (-9, 7, "right"), (-9, -14, "right"),
             (9, 21, "left"), (9, -28, "left"), (-9, 21, "right"), (-9, -28, "right")]
    placed = []
    for x, y, name in zip(xs, ys, names):
        for dx, dy, ha in tries:
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            color=INK_2, fontsize=9.5)
            box = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(b) for b in placed):
                break
            t.remove()
        else:  # every position was taken: keep the first one
            t = ax.annotate(name, (x, y), xytext=tries[0][:2], textcoords="offset points",
                            color=INK_2, fontsize=9.5)
            box = t.get_window_extent(renderer)
        placed.append(box)


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Loading and the few tables every plot needs
# --------------------------------------------------------------------------- #


def load_log(path: str | Path) -> pd.DataFrame:
    """Read the logbook written by measurements.log_evaluation()."""
    df = pd.read_json(path, lines=True)
    need = {"character", "line_id", "generation", "fitness", *B_COLS}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"logbook is missing columns: {sorted(missing)}")
    df["character"] = df["character"].astype(str)
    return df


def best_per_generation(df: pd.DataFrame) -> pd.DataFrame:
    """The best individual of every (character, line, generation)."""
    idx = df.groupby(["character", "line_id", "generation"])["fitness"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def winners(df: pd.DataFrame) -> pd.DataFrame:
    """The best individual each character ever found for each line."""
    idx = df.groupby(["character", "line_id"])["fitness"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Mean behaviour vector of the per-line best, for every (character, generation)."""
    return best_per_generation(df).groupby(["character", "generation"])[B_COLS].mean().reset_index()


# --------------------------------------------------------------------------- #
# Plots for an evolution run
# --------------------------------------------------------------------------- #


def plot_fitness(df: pd.DataFrame, out: Path, note: str = "") -> Path:
    best = best_per_generation(df).groupby(["character", "generation"])["fitness"].mean().reset_index()
    col = _colours(list(best.character.unique()))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.09, right=0.84, top=0.80, bottom=0.13)
    ends = []
    for ch, g in best.groupby("character"):
        ax.plot(g.generation, g.fitness, color=col[ch], label=ch)
        ax.plot(g.generation.iloc[-1], g.fitness.iloc[-1], "o", color=col[ch],
                markeredgecolor=SURFACE, markeredgewidth=2, markersize=9)
        ends.append((g.generation.iloc[-1], g.fitness.iloc[-1], ch))
    _end_labels(ax, ends)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness, mean over lines")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    if len(col) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.10), ncol=len(col), handlelength=1.4,
                  columnspacing=1.6, labelcolor=INK_2)
    _title(fig, "Does the search improve?", "Best fitness per generation, one line per character.", note)
    return _save(fig, out, "fitness.png")


def plot_fitness_terms(df: pd.DataFrame, out: Path, note: str = "") -> Path:
    terms = [("validity", "Validity V", "listenable and sensible"),
             ("wundt", "Wundt novelty W(N)", "new, but not absurd"),
             ("distinct", "Distinctiveness D", "unlike the other characters")]
    terms = [t for t in terms if t[0] in df.columns]
    best = best_per_generation(df).groupby(["character", "generation"])[[t[0] for t in terms]] \
        .mean().reset_index()
    col = _colours(list(best.character.unique()))
    fig, axes = plt.subplots(1, len(terms), figsize=(11, 4.2), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.70, bottom=0.14, wspace=0.22)
    for ax, (key, name, meaning) in zip(np.atleast_1d(axes), terms):
        for ch, g in best.groupby("character"):
            ax.plot(g.generation, g[key], color=col[ch], label=ch)
        ax.set_title(f"{name}\n", loc="left", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(0, 1.03, meaning, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")
        ax.set_xlabel("Generation")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(axis="x", visible=False)
    if len(col) > 1:
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 0.875), ncol=len(col),
                   handlelength=1.4, columnspacing=1.6, labelcolor=INK_2)
    _title(fig, "Which part of the fitness is doing the work?",
           "Each term for the best individual per generation. Each panel has its own scale.", note)
    return _save(fig, out, "fitness_terms.png")


def _pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain PCA with numpy. Returns (mean, 2 components, share of variance explained)."""
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    return mean, vt[:2], var[:2]


def plot_behaviour_map(df: pd.DataFrame, out: Path, note: str = "") -> Path:
    mean, comp, var = _pca2(df[B_COLS].to_numpy())
    project = lambda m: (np.asarray(m) - mean) @ comp.T
    col, mark = _colours(list(df.character.unique())), _markers(list(df.character.unique()))
    win, cen = winners(df), centroids(df)

    fig, ax = plt.subplots(figsize=(8, 6.4))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.79, bottom=0.10)
    for ch, g in df.groupby("character"):                       # every evaluation, faint
        p = project(g[B_COLS])
        ax.scatter(p[:, 0], p[:, 1], s=9, color=col[ch], alpha=0.16, linewidths=0)
    for ch, g in cen.groupby("character"):                      # path of the centroid
        p = project(g.sort_values("generation")[B_COLS])
        ax.plot(p[:, 0], p[:, 1], color=col[ch], linewidth=1.5, alpha=0.9)
    for ch, g in win.groupby("character"):                      # the winners, one per line
        p = project(g[B_COLS])
        ax.scatter(p[:, 0], p[:, 1], s=110, color=col[ch], marker=mark[ch], edgecolors=SURFACE,
                   linewidths=1.5, label=ch, zorder=3)
        ax.annotate(ch, (p[:, 0].mean(), p[:, 1].max()), xytext=(0, 12), textcoords="offset points",
                    ha="center", color=INK, fontsize=10.5, fontweight="bold", zorder=4)
    o = project(np.zeros((1, len(B_COLS))))[0]                  # the neutral render
    ax.scatter(*o, marker="+", s=160, color=INK, linewidths=1.8, zorder=4)
    ax.annotate("neutral", o, xytext=(9, -12), textcoords="offset points", color=INK_2, fontsize=10)

    ax.set_xlabel(f"Component 1 ({var[0]:.0%} of the variation)")
    ax.set_ylabel(f"Component 2 ({var[1]:.0%} of the variation)")
    ax.set_aspect("equal", adjustable="datalim")
    if len(col) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=len(col), handletextpad=0.3,
                  columnspacing=1.6, labelcolor=INK_2)
    _title(fig, "Where did each character end up?",
           "Large marks: the winner for each line. Thin line: path of the character's centroid.\n"
           "Faint dots: every evaluation. The six behaviour features are squeezed to 2-D with PCA.", note)
    return _save(fig, out, "behaviour_map.png")


def plot_profiles(df: pd.DataFrame, out: Path, note: str = "") -> Path:
    win = winners(df)
    chars = sorted(win.character.unique())
    col, mark = _colours(chars), _markers(chars)
    mean = win.groupby("character")[B_COLS].mean()
    lo = win.groupby("character")[B_COLS].min()
    hi = win.groupby("character")[B_COLS].max()

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.subplots_adjust(left=0.24, right=0.86, top=0.80, bottom=0.12)
    step = 0.5 / max(len(chars), 1)
    for r, name in enumerate(FEATURE_NAMES):
        for i, ch in enumerate(chars):
            y = r + (i - (len(chars) - 1) / 2) * step
            ax.plot([lo.loc[ch, f"b_{name}"], hi.loc[ch, f"b_{name}"]], [y, y], color=col[ch],
                    linewidth=1.5, alpha=0.45, solid_capstyle="round")
            ax.plot(mean.loc[ch, f"b_{name}"], y, mark[ch], color=col[ch], markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=ch if r == 0 else None)
    ax.axvline(0, color=AXIS, linewidth=1.2)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([f"{FEATURE_LABELS[n]}\n\u2190 {FEATURE_ENDS[n][0]}" for n in FEATURE_NAMES], color=INK_2)
    for r, name in enumerate(FEATURE_NAMES):
        ax.text(1.02, r, f"{FEATURE_ENDS[name][1]} \u2192", transform=ax.get_yaxis_transform(), va="center",
                ha="left", color=INK_2, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Distance from the neutral render (0), in typical steps of change")
    if len(chars) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.11), ncol=len(chars), handletextpad=0.3,
                  columnspacing=1.6, labelcolor=INK_2)
    _title(fig, "What does each character sound like?",
           "Mean of the winning performances per feature. The thin bar spans the lines.", note)
    return _save(fig, out, "profiles.png")


def plot_separation(df: pd.DataFrame, out: Path, note: str = "") -> Path | None:
    cen = centroids(df)
    if cen.character.nunique() < 2:
        print("separation.png skipped: needs at least two characters")
        return None
    rows = []
    for gen, g in cen.groupby("generation"):
        pts = g[B_COLS].to_numpy()
        if len(pts) >= 2:
            rows.append((gen, np.mean([np.linalg.norm(a - b) for a, b in combinations(pts, 2)])))
    sep = pd.DataFrame(rows, columns=["generation", "distance"])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.80, bottom=0.14)
    ax.plot(sep.generation, sep.distance, color=SERIES[0])
    ax.plot(sep.generation.iloc[-1], sep.distance.iloc[-1], "o", color=SERIES[0], markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{sep.distance.iloc[-1]:.2f}", (sep.generation.iloc[-1], sep.distance.iloc[-1]),
                xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean distance between character centroids")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    _title(fig, "Do the characters drift apart?",
           "Distance in the full six-feature behaviour space. Rising = more distinct characters.", note)
    return _save(fig, out, "separation.png")


def plot_run(log: str | Path, out: str | Path, note: str = "") -> list[Path]:
    df, out = load_log(log), Path(out)
    made = [plot_fitness(df, out, note), plot_fitness_terms(df, out, note),
            plot_behaviour_map(df, out, note), plot_profiles(df, out, note),
            plot_separation(df, out, note)]
    return [p for p in made if p]


# --------------------------------------------------------------------------- #
# Voice map: for a CSV made by `measurements.py measure`
# --------------------------------------------------------------------------- #


def plot_voices(csv_path: str | Path, out: str | Path) -> list[Path]:
    df, out = pd.read_csv(csv_path), Path(out)
    panels = [
        ("f0_median_hz", "formant_dispersion_hz", "voice_map_size.png", "How deep and how large is each voice?",
         "Lower left = deep voice from a large speaker. Upper right = high voice from a small speaker.",
         "Median pitch (Hz)", "Mean formant spacing (Hz)"),
        ("hnr_db", "jitter", "voice_map_quality.png", "How clear or rough is each voice?",
         "Left = breathy or husky (more noise). Up = rough or creaky (unsteady pitch).",
         "Harmonics-to-noise ratio (dB)", "Jitter (share of the pitch period)"),
    ]
    made = []
    for x, y, fname, title, sub, xl, yl in panels:
        d = df.dropna(subset=[x, y])
        if d.empty:
            print(f"{fname} skipped: no values for {x} / {y}")
            continue
        fig, ax = plt.subplots(figsize=(8, 5.6))
        fig.subplots_adjust(left=0.11, right=0.95, top=0.84, bottom=0.11)
        ax.scatter(d[x], d[y], s=80, color=SERIES[0], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.margins(0.15)
        _label_points(fig, ax, d[x].to_numpy(), d[y].to_numpy(),
                      [Path(str(f)).stem for f in d["file"]])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        _title(fig, title, sub)
        made.append(_save(fig, out, fname))
    return made


# --------------------------------------------------------------------------- #
# Demo: a simulated logbook, so the plots can be tried before any real run exists
# --------------------------------------------------------------------------- #


def simulate_log(path: str | Path, characters=("merchant", "priest", "soldier"), lines: int = 6,
                 generations: int = 8, population: int = 12, seed: int = 7) -> Path:
    """Write a made-up logbook with the same columns as a real one. NOT real results."""
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = {c: rng.normal(0, 1.3, len(FEATURE_NAMES)) for c in characters}
    rows = []
    for c in characters:
        for line in range(lines):
            line_shift = rng.normal(0, 0.25, len(FEATURE_NAMES))
            for gen in range(generations):
                progress = 1 - np.exp(-gen / 2.5)
                for ind in range(population):
                    b = progress * targets[c] + line_shift + rng.normal(0, 0.55 * (1.15 - progress), 6)
                    n = float(np.linalg.norm(b))
                    validity = float(np.clip(rng.normal(0.80 + 0.15 * progress, 0.10), 0, 1))
                    w = float(np.exp(-((n - 1.5) ** 2) / (2 * 0.75 ** 2)))
                    others = [targets[o] * progress for o in characters if o != c]
                    d = float(np.tanh(min(np.linalg.norm(b - o) for o in others)
                                      - np.linalg.norm(b - targets[c] * progress))) if gen else 0.0
                    rows.append({"run": "demo", "line_id": line, "character": c, "generation": gen,
                                 "individual": ind, "genome": "[]", "validity": validity, "novelty": n,
                                 "wundt": w, "distinct": d, "fitness": validity * (0.5 * w + 0.5 * d),
                                 **{f"b_{k}": float(v) for k, v in zip(FEATURE_NAMES, b)}})
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    print(f"wrote simulated logbook {path} ({len(rows)} evaluations)")
    return path


# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Dramatizer measurements.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="plots for an evolution run")
    r.add_argument("log", help="the .jsonl logbook")
    r.add_argument("--out", default="plots")
    v = sub.add_parser("voices", help="voice map from a measurements CSV")
    v.add_argument("csv")
    v.add_argument("--out", default="plots")
    d = sub.add_parser("demo", help="simulate a logbook and plot it")
    d.add_argument("--out", default="plots_demo")
    args = p.parse_args()
    if args.cmd == "run":
        plot_run(args.log, args.out)
    elif args.cmd == "voices":
        plot_voices(args.csv, args.out)
    else:
        log = simulate_log(Path(args.out) / "demo_logbook.jsonl")
        plot_run(log, args.out, note="SIMULATED DATA, not results")


if __name__ == "__main__":
    main()
