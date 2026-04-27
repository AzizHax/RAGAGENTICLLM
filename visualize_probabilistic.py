#!/usr/bin/env python3
"""
visualize_probabilistic.py — Plots pour le mémoire

Produit:
  1. Distribution des 5 classes PR par architecture (B1-B4)
  2. Courbe de calibration P(PR+) vs taux observé
  3. Comparaison rule-based vs probabiliste (confusion matrices)
  4. Histogramme P(PR+) avec séparation RA+/RA-
  5. Trajectoires temporelles patients
  6. Heatmap features → classes

Usage:
    python visualize_probabilistic.py --run-dir runs/b2
    python visualize_probabilistic.py --compare runs/b1 runs/b2 runs/b3 runs/b4
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CLASSES = ["PR_ABSENT", "PR_LATENT", "PR_REMISSION", "PR_MODERATE", "PR_SEVERE"]
CLASS_COLORS = {
    "PR_ABSENT": "#4CAF50",
    "PR_LATENT": "#FF9800",
    "PR_REMISSION": "#2196F3",
    "PR_MODERATE": "#F44336",
    "PR_SEVERE": "#9C27B0",
}
ARCH_COLORS = {"B1": "#1565C0", "B2": "#2E7D32", "B3": "#EF6C00", "B4": "#C62828"}


def load_run(run_dir: str):
    """Load assessments, decisions, and optionally ground truth."""
    rd = Path(run_dir)
    assessments, decisions, gt = [], [], {}

    af = rd / "assessments.jsonl"
    if af.exists():
        with open(af) as f:
            assessments = [json.loads(l) for l in f]

    df = rd / "decisions.jsonl"
    if df.exists():
        with open(df) as f:
            decisions = [json.loads(l) for l in f]

    # Try to find ground truth
    for gp in [rd / "ground_truth.json", rd.parent / "ground_truth_50p.json",
               Path("data/test_corpus/ground_truth_50p.json")]:
        if gp.exists():
            gt = json.load(open(gp))
            break

    return assessments, decisions, gt


# ════════════════════════════════════════════════════════════════
# PLOT 1: Distribution des classes PR
# ════════════════════════════════════════════════════════════════

def plot_class_distribution(decisions, gt, output_dir, tag=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Patient-level classes
    classes = [d.get("patient_predicted_class", "") or "?" for d in decisions]
    counts = Counter(classes)
    labels_sorted = [c for c in CLASSES if c in counts] + [c for c in counts if c not in CLASSES]
    vals = [counts.get(c, 0) for c in labels_sorted]
    colors = [CLASS_COLORS.get(c, "#999") for c in labels_sorted]

    bars = ax1.barh(labels_sorted, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Nombre de patients")
    ax1.set_title("Distribution des classes PR (patient-level)")
    ax1.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v} ({100*v/len(decisions):.0f}%)", va="center", fontsize=10)

    # P(PR+) histogram colored by GT
    probs = []
    colors_hist = []
    for d in decisions:
        p = d.get("patient_pr_positive_probability", 0)
        probs.append(p)
        pid = d["patient_id"]
        true = gt.get(pid, d["final_label"])
        colors_hist.append("#F44336" if true == "RA+" else "#4CAF50")

    # Split into RA+ and RA-
    probs_pos = [p for p, c in zip(probs, colors_hist) if c == "#F44336"]
    probs_neg = [p for p, c in zip(probs, colors_hist) if c == "#4CAF50"]

    bins = np.linspace(0, 1, 21)
    ax2.hist(probs_neg, bins=bins, alpha=0.7, color="#4CAF50", label=f"RA- (n={len(probs_neg)})", edgecolor="white")
    ax2.hist(probs_pos, bins=bins, alpha=0.7, color="#F44336", label=f"RA+ (n={len(probs_pos)})", edgecolor="white")
    ax2.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="Seuil 0.5")
    ax2.set_xlabel("P(PR+)")
    ax2.set_ylabel("Nombre de patients")
    ax2.set_title("Distribution P(PR+) par label réel")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"01_class_distribution{tag}.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PLOT 2: Courbe de calibration
# ════════════════════════════════════════════════════════════════

def plot_calibration(decisions, gt, output_dir, tag=""):
    if not gt:
        print("  [SKIP] Calibration: pas de ground truth")
        return

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    probs = []
    trues = []
    for d in decisions:
        pid = d["patient_id"]
        true = gt.get(pid)
        if true is None:
            continue
        probs.append(d.get("patient_pr_positive_probability", 0))
        trues.append(1 if true == "RA+" else 0)

    if not probs:
        plt.close(fig)
        return

    probs = np.array(probs)
    trues = np.array(trues)

    # Bin probabilities
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_true_rates = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1] + 1e-9)
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_true_rates.append(trues[mask].mean())
            bin_counts.append(mask.sum())

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibration parfaite")

    # Calibration curve
    ax.plot(bin_centers, bin_true_rates, "o-", color="#1565C0", linewidth=2,
            markersize=10, label="Modèle")

    # Annotate counts
    for x, y, n in zip(bin_centers, bin_true_rates, bin_counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(10, -10), fontsize=9, color="#666")

    ax.set_xlabel("P(PR+) prédite", fontsize=12)
    ax.set_ylabel("Proportion réelle de PR+", fontsize=12)
    ax.set_title("Courbe de calibration", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Brier score
    brier = np.mean((probs - trues) ** 2)
    ax.text(0.05, 0.90, f"Brier Score: {brier:.4f}", transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"02_calibration{tag}.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PLOT 3: Comparaison rule-based vs probabiliste
# ════════════════════════════════════════════════════════════════

def plot_rule_vs_prob(decisions, gt, output_dir, tag=""):
    if not gt:
        print("  [SKIP] Comparaison: pas de ground truth")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    for ax_idx, (method, get_pred) in enumerate([
        ("Rule-based (label)", lambda d: d["final_label"]),
        ("Probabiliste (P≥0.5)", lambda d: "RA+" if d.get("patient_pr_positive_probability", 0) >= 0.5 else "RA-"),
    ]):
        ax = axes[ax_idx]
        tp = fp = tn = fn = 0
        for d in decisions:
            true = gt.get(d["patient_id"])
            if true is None: continue
            pred = get_pred(d)
            if pred == "RA+" and true == "RA+": tp += 1
            elif pred == "RA+" and true == "RA-": fp += 1
            elif pred == "RA-" and true == "RA-": tn += 1
            else: fn += 1

        mat = np.array([[tn, fp], [fn, tp]])
        total = tp + fp + tn + fn
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        acc = (tp + tn) / total if total else 0

        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.flatten()) * 1.2)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred RA-", "Pred RA+"])
        ax.set_yticklabels(["True RA-", "True RA+"])
        ax.set_title(f"{method}\nF1={f1:.2f} Acc={acc:.2f} Prec={prec:.2f} Rec={rec:.2f}", fontsize=11)

        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=20, fontweight="bold", color=color)

    fig.suptitle("Rule-based vs Probabiliste", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"03_rule_vs_prob{tag}.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PLOT 4: Trajectoires temporelles
# ════════════════════════════════════════════════════════════════

def plot_trajectories(decisions, output_dir, tag=""):
    # Find patients with interesting trajectories (not all stable)
    interesting = []
    for d in decisions:
        trend = d.get("patient_temporal_trend", "stable")
        n_stays = d.get("n_stays", 0)
        if trend != "stable" and n_stays >= 2:
            interesting.append(d)
    if not interesting:
        interesting = decisions[:6]

    interesting = interesting[:6]  # Max 6 patients

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    axes = axes.flatten()

    for idx, d in enumerate(interesting):
        if idx >= 6:
            break
        ax = axes[idx]
        pid = d["patient_id"]
        trend = d.get("patient_temporal_trend", "?")
        stays = d.get("stay_details", [])

        # Per-stay class probabilities
        x_vals = list(range(1, len(stays) + 1))
        class_data = {c: [] for c in CLASSES}

        for s in stays:
            cp = s.get("class_probabilities", {})
            for c in CLASSES:
                class_data[c].append(cp.get(c, 0))

        # Stacked area
        bottom = np.zeros(len(x_vals))
        for c in CLASSES:
            vals = np.array(class_data[c])
            ax.fill_between(x_vals, bottom, bottom + vals,
                            alpha=0.7, color=CLASS_COLORS[c], label=c)
            bottom += vals

        ax.set_title(f"{pid} ({trend})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Séjour")
        ax.set_ylabel("P(classe)")
        ax.set_ylim(0, 1)
        ax.set_xlim(1, max(len(stays), 2))

    # Hide unused axes
    for idx in range(len(interesting), 6):
        axes[idx].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Trajectoires temporelles — Distribution des classes par séjour",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(Path(output_dir) / f"04_trajectories{tag}.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PLOT 5: Comparaison multi-architecture
# ════════════════════════════════════════════════════════════════

def plot_multi_arch_comparison(run_dirs, output_dir):
    """Compare class distributions across B1-B4."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    arch_data = {}
    arch_metrics = {}

    for rd in run_dirs:
        rd = Path(rd)
        arch = rd.name.upper()
        _, decisions, gt = load_run(str(rd))
        if not decisions:
            continue

        # Class distribution
        classes = [d.get("patient_predicted_class", "") or "?" for d in decisions]
        arch_data[arch] = Counter(classes)

        # Metrics
        if gt:
            tp = fp = tn = fn = 0
            for d in decisions:
                true = gt.get(d["patient_id"])
                if true is None: continue
                pred = d["final_label"]
                if pred == "RA+" and true == "RA+": tp += 1
                elif pred == "RA+" and true == "RA-": fp += 1
                elif pred == "RA-" and true == "RA-": tn += 1
                else: fn += 1
            total = tp + fp + tn + fn
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            acc = (tp + tn) / total if total else 0
            arch_metrics[arch] = {"f1": f1, "acc": acc, "prec": prec, "rec": rec}

    if not arch_data:
        print("  [SKIP] Multi-arch: pas de données")
        plt.close(fig)
        return

    # Plot 1: Grouped bar chart of class distributions
    ax = axes[0]
    archs = sorted(arch_data.keys())
    x = np.arange(len(CLASSES))
    width = 0.8 / len(archs)

    for i, arch in enumerate(archs):
        counts = [arch_data[arch].get(c, 0) for c in CLASSES]
        color = ARCH_COLORS.get(arch, "#999")
        ax.bar(x + i * width, counts, width, label=arch, color=color, alpha=0.8, edgecolor="white")

    ax.set_xticks(x + width * (len(archs) - 1) / 2)
    ax.set_xticklabels([c.replace("PR_", "") for c in CLASSES], fontsize=9)
    ax.set_ylabel("Nombre de patients")
    ax.set_title("Distribution des classes par architecture")
    ax.legend()

    # Plot 2: Metrics comparison
    ax = axes[1]
    if arch_metrics:
        metrics_names = ["f1", "acc", "prec", "rec"]
        metrics_labels = ["F1-Score", "Accuracy", "Precision", "Recall"]
        x = np.arange(len(metrics_names))
        width = 0.8 / len(arch_metrics)

        for i, (arch, m) in enumerate(sorted(arch_metrics.items())):
            vals = [m[k] for k in metrics_names]
            color = ARCH_COLORS.get(arch, "#999")
            bars = ax.bar(x + i * width, vals, width, label=arch, color=color, alpha=0.8, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x + width * (len(arch_metrics) - 1) / 2)
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 1.15)
        ax.set_title("Métriques par architecture")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Pas de ground truth", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Métriques (non disponibles)")

    fig.suptitle("Comparaison multi-architecture", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "05_multi_arch_comparison.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PLOT 6: Tableau métriques rule-based vs probabiliste
# ════════════════════════════════════════════════════════════════

def plot_metrics_table(decisions, gt, output_dir, tag=""):
    if not gt:
        print("  [SKIP] Metrics table: pas de ground truth")
        return

    methods = {
        "Rule-based": lambda d: d["final_label"],
        "Prob (P≥0.5)": lambda d: "RA+" if d.get("patient_pr_positive_probability", 0) >= 0.5 else "RA-",
        "Prob (P≥0.6)": lambda d: "RA+" if d.get("patient_pr_positive_probability", 0) >= 0.6 else "RA-",
        "Prob (P≥0.7)": lambda d: "RA+" if d.get("patient_pr_positive_probability", 0) >= 0.7 else "RA-",
    }

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)
    ax.axis("off")

    headers = ["Méthode", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"]
    rows = []

    for method_name, get_pred in methods.items():
        tp = fp = tn = fn = 0
        for d in decisions:
            true = gt.get(d["patient_id"])
            if true is None: continue
            pred = get_pred(d)
            if pred == "RA+" and true == "RA+": tp += 1
            elif pred == "RA+" and true == "RA-": fp += 1
            elif pred == "RA-" and true == "RA-": tn += 1
            else: fn += 1
        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        rows.append([method_name, tp, fp, tn, fn,
                      f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])

    table = ax.table(cellText=rows, colLabels=headers,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best F1
    f1_vals = [float(r[-1]) for r in rows]
    best_idx = f1_vals.index(max(f1_vals))
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor("#E8F5E9")

    ax.set_title("Comparaison Rule-based vs Probabiliste (différents seuils)",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"06_metrics_table{tag}.png")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualisation probabiliste pour le mémoire")
    parser.add_argument("--run-dir", default="runs/b2", help="Dossier de résultats")
    parser.add_argument("--compare", nargs="+", help="Comparer plusieurs architectures (ex: runs/b1 runs/b2)")
    parser.add_argument("--output", default=None, help="Dossier de sortie (défaut: {run-dir}/plots)")
    parser.add_argument("--gt", default="data/test_corpus/ground_truth_50p.json", help="Ground truth")
    args = parser.parse_args()

    output_dir = args.output or str(Path(args.run_dir) / "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Visualisation Probabiliste")
    print("=" * 60)

    # Load data
    assessments, decisions, gt = load_run(args.run_dir)
    if not gt and Path(args.gt).exists():
        gt = json.load(open(args.gt))

    print(f"  Run dir:     {args.run_dir}")
    print(f"  Assessments: {len(assessments)}")
    print(f"  Decisions:   {len(decisions)}")
    print(f"  Ground truth:{len(gt)} patients")
    print(f"  Output:      {output_dir}")

    # Single-run plots
    print("\n[1/6] Distribution des classes...")
    plot_class_distribution(decisions, gt, output_dir)

    print("[2/6] Courbe de calibration...")
    plot_calibration(decisions, gt, output_dir)

    print("[3/6] Rule-based vs Probabiliste...")
    plot_rule_vs_prob(decisions, gt, output_dir)

    print("[4/6] Trajectoires temporelles...")
    plot_trajectories(decisions, output_dir)

    print("[5/6] Tableau métriques comparatif...")
    plot_metrics_table(decisions, gt, output_dir)

    # Multi-arch comparison
    if args.compare:
        print("[6/6] Comparaison multi-architecture...")
        plot_multi_arch_comparison(args.compare, output_dir)
    else:
        print("[6/6] Comparaison multi-arch: --compare non spécifié, skip")

    print(f"\nPlots sauvegardés dans {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
