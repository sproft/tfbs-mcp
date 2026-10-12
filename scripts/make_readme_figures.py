"""Draw the README's task figure (docs/figure1.png) and the layer diagram
(docs/architecture.png) for tfbs-mcp. Usage: python make_readme_figures.py <docs dir>"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(sys.argv[1])
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

LQ, RQ, ARR, MINUS = "“", "”", "→", "−"


def rbox(ax, x, y, w, h, text, face, edge="#444444", size=10.5, weight="normal",
         color="black", lw=1.0, style="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=lw, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight=weight, color=color, style=style, linespacing=1.25)


def arrow(ax, x1, y, x2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=13,
                                 color=color, linewidth=1.3))


def task_figure():
    rows = [
        ("Does a variant change\nhow strongly my TF binds?",
         f"{LQ}Does demo_chr8:830:T>G weaken\nNKX2-1 binding? Use the released\nFLANK model.{RQ}",
         "load the model, cut the reference\nand variant windows from the\ngenome, score both",
         f"1.07 {ARR} {MINUS}0.54 log2 fold change:\nabout 3x weaker binding"),
        ("Can a model learn my TF\nfrom ChIP-seq peaks?",
         f"{LQ}Train an NKX2-1 classifier on\nthe ChIP-Atlas peaks in\npeaks.bed.{RQ}",
         "cut peak windows and GC-matched\nbackground, split, train here\nor on a SLURM cluster",
         "a trained model and its\nvalidation accuracy and AUC"),
        ("Which bases drive\na prediction?",
         f"{LQ}Show what drives the score of\nGGGCTCCTCACTTGACCTTTGCTG.{RQ}",
         "attributions, in-silico\nmutagenesis, motif insertion",
         "the positions that matter\nand the effect of the motif"),
        ("How strongly does the\nprotein bind this DNA?",
         f"{LQ}Fetch PDB 3RKQ and compute\nits binding energy with FoldX.{RQ}",
         "fetch from PDB or AlphaFold DB,\nrepair with FoldX, compute the\nenergy (or scan a region)",
         "interaction energy in kcal/mol,\na bigWig track along the genome"),
        ("Does my model beat\na motif scan?",
         f"{LQ}Validate my model against\nReMap and ClinVar, head to head\nwith FIMO and FoldX.{RQ}",
         "score the same peaks or variants\nwith the network, FIMO and FoldX",
         "ROC-AUC of each method,\nwith plots and tables"),
    ]
    heads = [("You want to know", "#fdebd3"), ("You type in Claude's chat", "#e3eefc"),
             ("Claude runs tfbs-mcp tools", "#dff3e4"), ("You get", "#f3e5f5")]
    W, H = 14.2, 7.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ws = [3.0, 3.15, 3.15, 3.1]
    xs = [0.1]
    for w in ws[:-1]:
        xs.append(xs[-1] + w + 0.4)
    top = H - 0.15
    for (label, face), x, w in zip(heads, xs, ws):
        rbox(ax, x, top - 0.55, w, 0.55, label, face, size=12.5, weight="bold", edge="#888888")
    rh, gap = 1.04, 0.16
    band_bottom = top - 0.75 - 5 * rh - 4 * gap - 0.55
    ax.add_patch(FancyBboxPatch((xs[2] - 0.08, band_bottom), ws[2] + 0.16,
                                top - 0.62 - band_bottom,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=1.2, edgecolor="#2e7d32", facecolor="#f4fbf5",
                                linestyle="--"))
    y = top - 0.75 - rh
    for goal, prompt, steps, result in rows:
        rbox(ax, xs[0], y, ws[0], rh, goal, "#fff6ea", size=11, weight="bold")
        rbox(ax, xs[1], y, ws[1], rh, prompt, "#f2f7fe", size=10, style="italic")
        rbox(ax, xs[2], y, ws[2], rh, steps, "#ffffff", size=10.5)
        rbox(ax, xs[3], y, ws[3], rh, result, "#fbf3fc", size=10.5, weight="bold")
        for i in range(3):
            arrow(ax, xs[i] + ws[i] + 0.06, y + rh / 2, xs[i + 1] - 0.06)
        y -= rh + gap
    ax.text(xs[2] + ws[2] / 2, band_bottom + 0.27,
            "Claude picks the tools and passes\neach result to the next step",
            ha="center", va="center", fontsize=10, color="#2e7d32", style="italic")
    ax.text(W / 2, band_bottom - 0.3,
            "Runs on your computer with your own genome, peak, motif and ClinVar files; "
            "long jobs can go to a SLURM cluster.",
            ha="center", va="center", fontsize=11, color="#333333")
    fig.savefig(OUT / "figure1.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(8.95, 4.75))
    ax.set_xlim(0, 8.95)
    ax.set_ylim(0, 4.75)
    ax.axis("off")

    def box(x, y, w, h, label, face, size=9.5, weight="normal", title=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.18",
                                    linewidth=1.0, edgecolor="black", facecolor=face))
        if title:
            ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=size,
                    fontweight="bold")
            ax.text(x + w / 2, y + (h - 0.4) / 2 + 0.05, label, ha="center", va="center",
                    fontsize=size - 1)
        else:
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size,
                    fontweight=weight)

    box(2.3, 4.08, 4.2, 0.62, "LLM client\n(Claude Code, Claude Desktop, other MCP clients)",
        "#f0f0f0", size=9.2)
    sx, sy, sw, sh = 2.3, 0.1, 4.2, 3.6
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.06,rounding_size=0.18",
                                linewidth=1.3, edgecolor="black", facecolor="#fafafa"))
    ax.text(sx + sw / 2, sy + sh - 0.08, "tfbs-mcp server  (FastMCP, Python)", ha="center",
            va="top", fontsize=10.5, fontweight="bold")
    cols, cw, lh, rows = [2.42, 4.63], 1.75, 0.78, [2.45, 1.33, 0.21]
    layers = [("Inference", 10), ("SLURM management", 5), ("Training", 4),
              ("Data pipeline", 7), ("Structure-based", 3), ("Validation &\ncomparison", 4)]
    for i, (name, n) in enumerate(layers):
        box(cols[i % 2], rows[i // 2], cw, lh, f"{name}\n{n} tools", "#eaf3ff", size=9.2,
            weight="bold")
    rx, rw, lx, lw = 7.2, 1.62, 0.08, 1.58
    box(rx, rows[0] - 0.05, rw, lh + 0.1, "(optional)\nSLURM, GPU nodes", "#eef5ff", size=9,
        title="HPC cluster")
    box(rx, rows[2] - 0.05, rw, rows[1] + lh - rows[2] + 0.1,
        "(user-downloaded)\ngenome FASTA,\npeak BEDs (ChIP-Atlas,\nReMap), JASPAR motif,\n"
        "ClinVar VCF", "#fff5e8", size=9, title="Local reference files")
    box(lx, rows[2] - 0.15, lw, lh + 0.42, "PDB, AlphaFold DB\n(downloaded);\nFoldX (local binary)",
        "#f0fbe9", size=9, title="Structure sources")

    def harrow(x1, x2, yy, label):
        ax.add_patch(FancyArrowPatch((x1, yy), (x2, yy), arrowstyle="-|>", mutation_scale=12,
                                     color="black", linewidth=1.1))
        ax.text((x1 + x2) / 2, yy + 0.06, label, fontsize=8.5, ha="center", va="bottom",
                style="italic")

    ax.add_patch(FancyArrowPatch((4.4, 4.08), (4.4, 3.77), arrowstyle="-|>", mutation_scale=12,
                                 color="black", linewidth=1.1))
    ax.text(4.52, 3.925, "MCP / stdio", fontsize=9, ha="left", va="center", style="italic")

    def mid(r):
        return rows[r] + lh / 2

    harrow(cols[1] + cw, rx, mid(0), "jobs")
    harrow(cols[1] + cw, rx, mid(1), "reads")
    harrow(cols[1] + cw, rx, mid(2), "reads")
    harrow(cols[0], lx + lw, mid(2), "fetches,\nruns")
    fig.savefig(OUT / "architecture.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


task_figure()
architecture_figure()
print("wrote", OUT / "figure1.png", OUT / "architecture.png")
