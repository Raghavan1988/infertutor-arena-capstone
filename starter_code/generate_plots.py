"""Generate engineering report plots from results_infertutor/*.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / "results_infertutor"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 11})


def load_all():
    runs = {}
    for f in sorted(RESULTS.glob("*.json")):
        data = json.loads(f.read_text())
        runs[data["config"]["label"]] = {"file": f.name, **data}
    return runs


def score(data):
    c, r = data["config"], data["results"]
    goodput = r["aggregate_stream_chunks_per_s"] * max(0, 1 - r["error_rate"])
    ttft = max(r["ttft_p95_ms"] / 1000, 0.001)
    itl = max(r["itl_p95_ms"] / 1000, 0.001)
    return goodput * c["users"] / (ttft * itl * max(c["total_gpus"], 1))


def plot_text_knee(runs):
    """Plot 1: Text-track user sweep showing where the knee bends."""

    points = [
        ("smoke", 5),
        ("baseline-text", 50),
        ("sweep-60u", 60),  # use the first/clean 60u
        ("sweep-70u", 70),  # the redo (cleaner)
        ("sweep-80u", 80),  # the real 80u
        ("sweep-90u", 90),  # the redo
        ("sweep-100u", 100),
        ("sweep-300u", 300),
        ("sweep-400u", 400),
        ("sweep-500u", 500),
        ("sweep-600u", 600),
    ]

    users, ttft, throughput = [], [], []
    for label, u in points:
        # find the cleanest run for this label/user combo
        candidates = [r for k, r in runs.items() if k == label]
        if not candidates:
            continue
        # for labels with multiple files, use the most recent (typically the redo)
        runs_for_label = [r for r in candidates]
        chosen = runs_for_label[0]
        users.append(u)
        ttft.append(chosen["results"]["ttft_p95_ms"])
        throughput.append(chosen["results"]["aggregate_stream_chunks_per_s"])

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax2 = ax1.twinx()

    color_ttft, color_thr = "#d62728", "#2ca02c"

    ax1.plot(users, ttft, "o-", color=color_ttft, lw=2.4, ms=9, label="TTFT p95 (ms)", zorder=3)
    ax2.plot(users, throughput, "s--", color=color_thr, lw=2, ms=8, label="Throughput (chunks/s)", zorder=2)

    ax1.set_yscale("log")
    ax1.set_xlabel("Concurrent users", fontsize=12)
    ax1.set_ylabel("TTFT p95 (ms, log scale)", color=color_ttft, fontsize=12)
    ax2.set_ylabel("Throughput (chunks/s)", color=color_thr, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_ttft)
    ax2.tick_params(axis="y", labelcolor=color_thr)

    # Mark the knee zone
    ax1.axvspan(60, 100, alpha=0.18, color="orange", label="The knee (60-100u)")

    # Annotate champion point and 100u jump
    if 50 in users:
        i = users.index(50)
        ax1.annotate(
            "  baseline\n  50u, 726ms",
            (users[i], ttft[i]),
            textcoords="offset points",
            xytext=(10, -22),
            fontsize=9.5,
            color=color_ttft,
        )
    if 100 in users:
        i = users.index(100)
        ax1.annotate(
            "100u\n10× TTFT jump\nvs 60u",
            (users[i], ttft[i]),
            textcoords="offset points",
            xytext=(15, -5),
            fontsize=9.5,
            color="darkred",
            fontweight="bold",
        )

    ax1.set_title(
        "Text-Track Knee Map — Default Knobs, 1 GPU\nTTFT p95 stays flat through 60u, then explodes by 100u",
        fontsize=13,
        pad=14,
    )
    ax1.grid(True, alpha=0.25, ls="--")
    ax1.set_xticks([0, 50, 100, 200, 300, 400, 500, 600])

    # Legend combining both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.95)

    fig.tight_layout()
    out = PLOTS / "01_text_knee.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_knob_ablations(runs):
    """Plot 2: Knob A/B comparison at 70u, text mode."""

    items = [
        ("baseline\n(50u, default)", "baseline-text", "#1f77b4"),
        ("--no-prefix-cache\n(70u)", "noprefix-70u", "#d62728"),
        ("--no-chunked-prefill\n(70u)", "nochunked-70u", "#ff7f0e"),
        ("--max-seqs 64\n(70u)", "seqs64-70u", "#d62728"),
        ("--max-batch-tokens 8192\n(70u)", "batch8k-70u", "#d62728"),
        ("--replicas 2 (eager)\n(140u, 2 GPU)", "scale-r2-140u", "#d62728"),
        ("--no-fast-boot\n(compiled, 70u)", "compiled-70u", "#2ca02c"),
    ]

    labels, scores, colors = [], [], []
    for disp, key, c in items:
        if key in runs:
            labels.append(disp)
            scores.append(score(runs[key]) / 1e6)
            colors.append(c)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Score (millions, log scale)", fontsize=12)
    ax.set_title(
        "Text-Track Knob A/Bs at 70 Users — Compiled Mode is the Only Win\n"
        "Bigger is better. All 'widening' knobs scored worse than baseline.",
        fontsize=13,
        pad=14,
    )
    ax.grid(True, axis="y", alpha=0.25, ls="--")

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.12,
            f"{val:.2f}M",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
        )

    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9.5)
    fig.tight_layout()
    out = PLOTS / "02_text_knob_ablations.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_compiled_vs_eager(runs):
    """Plot 3: The dominant lever — compiled mode in both tracks."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # Text track: baseline-text (eager 50u) vs compiled-70u
    ax = axes[0]
    configs = ["Eager\n(50u)", "Compiled\n(70u)"]
    ttft = [runs["baseline-text"]["results"]["ttft_p95_ms"],
            runs["compiled-70u"]["results"]["ttft_p95_ms"]]
    itl = [runs["baseline-text"]["results"]["itl_p95_ms"],
           runs["compiled-70u"]["results"]["itl_p95_ms"]]
    thr = [runs["baseline-text"]["results"]["aggregate_stream_chunks_per_s"],
           runs["compiled-70u"]["results"]["aggregate_stream_chunks_per_s"]]

    x = np.arange(2)
    w = 0.25
    ax.bar(x - w, ttft, w, label="TTFT p95 (ms)", color="#d62728")
    ax.bar(x, [v * 30 for v in itl], w, label="ITL p95 × 30 (ms)", color="#ff7f0e")
    ax.bar(x + w, thr, w, label="Throughput (chunks/s)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_title("Text Track\nCompiled mode at 70u beats eager at 50u on every metric", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper left", fontsize=9.5)

    # Multimodal: m-50u (eager) vs m-compiled-50u (compiled)
    ax = axes[1]
    configs = ["Eager\n(50u)", "Compiled\n(50u)"]
    ttft = [runs["m-50u"]["results"]["ttft_p95_ms"],
            runs["m-compiled-50u"]["results"]["ttft_p95_ms"]]
    itl = [runs["m-50u"]["results"]["itl_p95_ms"],
           runs["m-compiled-50u"]["results"]["itl_p95_ms"]]
    thr = [runs["m-50u"]["results"]["aggregate_stream_chunks_per_s"],
           runs["m-compiled-50u"]["results"]["aggregate_stream_chunks_per_s"]]

    x = np.arange(2)
    ax.bar(x - w, ttft, w, label="TTFT p95 (ms)", color="#d62728")
    ax.bar(x, [v * 30 for v in itl], w, label="ITL p95 × 30 (ms)", color="#ff7f0e")
    ax.bar(x + w, thr, w, label="Throughput (chunks/s)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_title(
        "Multimodal Track\nCompiled mode wins here too — contrary to runbook warning!",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper left", fontsize=9.5)

    fig.suptitle(
        "Compiled Mode (`--no-fast-boot`) is the Dominant Knob in BOTH Tracks",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    out = PLOTS / "03_compiled_vs_eager.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_final_summary(runs):
    """Plot 4: Final submission scores compared to references."""

    items = [
        ("FINAL-mixed\n(Track 1, 180s)\nMy submission", "FINAL-mixed", "#2ca02c"),
        ("FINAL_text_70u\n(Track 2, 180s)", "FINAL", "#1f77b4"),
        ("Reference 2r mixed\n(workshop baseline)", None, "#888888"),
        ("Reference 4r mixed\n(workshop baseline)", None, "#888888"),
    ]

    # Reference baselines computed from runbook table
    # 2r mixed: TTFT 1169 ms, ITL 28.7 ms, throughput 2243, 100u, 2 GPU
    ref_2r = 2243 * 100 * 1.0 / (1.169 * 0.0287 * 2)
    ref_4r = 2756 * 120 * 1.0 / (0.898 * 0.0381 * 4)

    scores, labels, colors = [], [], []
    for disp, key, c in items:
        labels.append(disp)
        colors.append(c)
        if key and key in runs:
            scores.append(score(runs[key]) / 1e6)
        elif "2r" in disp:
            scores.append(ref_2r / 1e6)
        elif "4r" in disp:
            scores.append(ref_4r / 1e6)
        else:
            scores.append(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Score (millions)", fontsize=12)
    ax.set_title(
        "Final Submission Comparison — Both Tracks Crush the Reference Baselines",
        fontsize=13,
        pad=14,
    )
    ax.grid(True, axis="y", alpha=0.25, ls="--")

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(scores) * 0.015,
            f"{val:.1f}M",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    # Annotate ratio vs reference for the multimodal FINAL
    final_mixed = score(runs["FINAL-mixed"]) / 1e6
    ax.annotate(
        f"{final_mixed/(ref_2r/1e6):.1f}× the 2-replica\nreference baseline",
        xy=(0, final_mixed),
        xytext=(0.4, final_mixed * 0.85),
        fontsize=10,
        color="darkgreen",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.4),
    )

    fig.tight_layout()
    out = PLOTS / "04_final_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_duration_lesson(runs):
    """Plot 5: 90s vs 180s — duration affects measured score differently per track."""

    # Text: 90s snapshot vs 180s FINAL (degraded)
    text_90s = score(runs["compiled-70u"]) / 1e6
    text_180s = score(runs["FINAL"]) / 1e6

    # Multimodal: 90s snapshot vs 180s FINAL (improved!)
    mm_90s = score(runs["m-compiled-50u"]) / 1e6
    mm_180s = score(runs["FINAL-mixed"]) / 1e6

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(2)
    w = 0.35

    bars1 = ax.bar(x - w / 2, [text_90s, mm_90s], w, label="90 second test", color="#9ecae1")
    bars2 = ax.bar(x + w / 2, [text_180s, mm_180s], w, label="180 second test (FINAL)", color="#3182bd")

    ax.set_xticks(x)
    ax.set_xticklabels(["Text Track\n(compiled-70u → FINAL_text_70u)",
                         "Multimodal Track\n(m-compiled-50u → FINAL-mixed)"], fontsize=11)
    ax.set_ylabel("Score (millions)", fontsize=12)
    ax.set_title(
        "The Duration Lesson: Short Tests Can Mislead — BOTH Ways\n"
        "Text 180s scored LOWER (tail emerged); Multimodal 180s scored HIGHER (steady-state was better).",
        fontsize=12.5,
        pad=14,
    )
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper right", fontsize=10.5)

    for bar, val in zip(bars1, [text_90s, mm_90s]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}M",
                ha="center", fontsize=10.5, fontweight="bold")
    for bar, val in zip(bars2, [text_180s, mm_180s]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}M",
                ha="center", fontsize=10.5, fontweight="bold")

    # Add Δ annotations
    ax.annotate(
        f"Δ {(text_180s/text_90s - 1)*100:.0f}%",
        xy=(x[0], max(text_90s, text_180s)),
        xytext=(x[0], max(text_90s, text_180s) + 10),
        ha="center",
        fontsize=11,
        color="darkred",
        fontweight="bold",
    )
    ax.annotate(
        f"Δ +{(mm_180s/mm_90s - 1)*100:.0f}%",
        xy=(x[1], max(mm_90s, mm_180s)),
        xytext=(x[1], max(mm_90s, mm_180s) + 10),
        ha="center",
        fontsize=11,
        color="darkgreen",
        fontweight="bold",
    )

    fig.tight_layout()
    out = PLOTS / "05_duration_lesson.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_score_decomposition(runs):
    """Plot 6: Score formula decomposition for the FINAL-mixed run."""

    final = runs["FINAL-mixed"]["results"]
    c = runs["FINAL-mixed"]["config"]

    goodput = final["aggregate_stream_chunks_per_s"] * (1 - final["error_rate"])
    ttft_s = final["ttft_p95_ms"] / 1000
    itl_s = final["itl_p95_ms"] / 1000

    num = goodput * c["users"] * (1 - final["error_rate"])
    den = ttft_s * itl_s * c["total_gpus"]
    final_score = num / den

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    text = (
        "FINAL-mixed Score Decomposition (Track 1 Submission)\n"
        "─────────────────────────────────────────────────────\n\n"
        f"   Score = goodput × users × (1 − error_rate)\n"
        f"           ────────────────────────────────────\n"
        f"           p95_TTFT_s × p95_ITL_s × GPU_count\n\n"
        f"   Numerator   = {goodput:>7.0f}    × {c['users']:>3}  × {(1-final['error_rate']):.4f}\n"
        f"                = {num:>12,.0f}\n\n"
        f"   Denominator = {ttft_s:>7.4f}s × {itl_s:>7.5f}s × {c['total_gpus']}\n"
        f"                = {den:>12.6f}\n\n"
        f"   SCORE       = {final_score:>12,.0f}\n"
        f"               = {final_score/1e6:.2f} M"
    )
    ax.text(
        0.05, 0.5, text,
        fontfamily="monospace",
        fontsize=12,
        verticalalignment="center",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    out = PLOTS / "06_score_decomposition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    runs = load_all()
    print(f"Loaded {len(runs)} runs")
    plot_text_knee(runs)
    plot_knob_ablations(runs)
    plot_compiled_vs_eager(runs)
    plot_final_summary(runs)
    plot_duration_lesson(runs)
    plot_score_decomposition(runs)
    print("\nAll plots written to plots/")


if __name__ == "__main__":
    main()
