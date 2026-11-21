"""Reproduce every metric and figure reported in the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("images")
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 20
PAPER_ACCURACY = 97.0
PAPER_FEATURES = ("worst area", "worst smoothness", "mean texture")

SURFACE, INK, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
MALIGNANT_COLOR, BENIGN_COLOR = "#eb6834", "#2a78d6"
ALL_FEATURES_COLOR, PAPER_FEATURES_COLOR = "#6250d6", "#1baf7a"

SCORERS = {
    "accuracy": "accuracy",
    "malignant_recall": make_scorer(recall_score, pos_label=0),
    "benign_recall": make_scorer(recall_score, pos_label=1),
    "roc_auc": "roc_auc",
}


def build_model() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))


def split(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def evaluate_split(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, np.ndarray, float]:
    X_train, X_test, y_train, y_test = split(X, y)
    model = build_model().fit(X_train, y_train)
    confusion = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, confusion, auc


def evaluate_cv(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    scores = cross_validate(build_model(), X, y, cv=cv, scoring=SCORERS)
    return {name: scores[f"test_{name}"] for name in SCORERS}


def print_results(name: str, confusion: np.ndarray, cv: dict[str, np.ndarray]) -> None:
    correct = np.trace(confusion)
    print(f"\n{name}")
    print(f"  Test split: {correct}/{confusion.sum()} correct")
    print(f"  Confusion matrix: {confusion.tolist()}")
    for metric, values in cv.items():
        print(f"  CV {metric}: {values.mean():.4f} +/- {values.std():.4f}")


def style_axes() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Arial", "sans-serif"],
            "font.size": 11,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_distributions(X: np.ndarray, y: np.ndarray, indices: list[int]) -> None:
    labels = (
        ("Worst area", "Area of largest nuclei"),
        ("Worst smoothness", "Local variation in radius"),
        ("Mean texture", "Variation in gray levels"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, index, (title, xlabel) in zip(axes, indices, labels):
        bins = np.linspace(X[:, index].min(), X[:, index].max(), 29)
        for target, color, label in (
            (1, BENIGN_COLOR, f"Benign ({(y == 1).sum()})"),
            (0, MALIGNANT_COLOR, f"Malignant ({(y == 0).sum()})"),
        ):
            ax.hist(
                X[y == target, index], bins=bins, color=color, alpha=0.62,
                edgecolor=SURFACE, linewidth=0.7, label=label,
            )
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Samples")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper right", ncol=2, frameon=False,
        bbox_to_anchor=(0.99, 1.0),
    )
    fig.suptitle(
        "The paper's three features, split by diagnosis",
        x=0.012, ha="left", fontweight="bold", fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=170)
    plt.close(fig)


def draw_confusion(ax: plt.Axes, confusion: np.ndarray, title: str) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#eef5fd", "#86b6ef", "#256abf"]
    )
    notes = {
        (0, 0): "cancer caught",
        (0, 1): "missed cancer",
        (1, 0): "false alarm",
        (1, 1): "correctly cleared",
    }
    ax.imshow(confusion, cmap=cmap, vmin=0, vmax=75)
    correct, total = np.trace(confusion), confusion.sum()
    ax.set_title(
        f"{title}: {correct}/{total} correct ({correct / total:.1%})",
        loc="left", fontweight="bold", fontsize=12, pad=10,
    )
    ax.set_xticks([0, 1], ["Malignant", "Benign"])
    ax.set_yticks([0, 1], ["Malignant", "Benign"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), note in notes.items():
        color = "#ffffff" if confusion[row, col] > 45 else INK
        ax.text(col, row - 0.08, str(confusion[row, col]), ha="center", va="center",
                fontsize=24, fontweight="bold", color=color)
        ax.text(col, row + 0.23, note, ha="center", va="center", fontsize=10,
                color=color)


def plot_confusions(all_features: np.ndarray, paper: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
    draw_confusion(axes[0], all_features, "All 30 features")
    draw_confusion(axes[1], paper, "Paper's 3 features")
    fig.suptitle(
        "Test-set results on the same 114 samples",
        x=0.012, ha="left", fontweight="bold", fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=170)
    plt.close(fig)


def plot_comparison(all_cv: dict[str, np.ndarray], paper_cv: dict[str, np.ndarray]) -> None:
    rows = (
        ("accuracy", "Accuracy"),
        ("malignant_recall", "Malignant recall\n(cancers caught)"),
        ("benign_recall", "Benign recall\n(benign cleared)"),
    )
    series = (
        (0.15, all_cv, ALL_FEATURES_COLOR, "All 30 features", "o"),
        (-0.15, paper_cv, PAPER_FEATURES_COLOR, "Paper's 3 features", "s"),
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for offset, scores, color, label, marker in series:
        for position, (metric, _) in enumerate(rows):
            y_pos = 2 - position + offset
            mean, std = scores[metric].mean() * 100, scores[metric].std() * 100
            ax.plot([mean - std, min(mean + std, 100)], [y_pos, y_pos], color=color,
                    lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(mean, y_pos, s=70, color=color, marker=marker, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3, label=label if position == 0 else None)
            ax.text(mean, y_pos + 0.12, f"{mean:.1f}%", ha="center", va="bottom",
                    fontsize=10)
    ax.plot([PAPER_ACCURACY] * 2, [1.72, 2.42], color=INK_SECONDARY, lw=1.4,
            ls=(0, (3, 2)))
    ax.text(PAPER_ACCURACY + 0.08, 1.67, f"paper: {PAPER_ACCURACY:.1f}%", ha="left",
            va="top", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks([2, 1, 0], [label for _, label in rows])
    ax.set_ylim(-0.5, 2.62)
    ax.set_xlim(86, 100.4)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean across 100 cross-validation folds; lines show ±1 SD (axis starts at 86%)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Cross-validated comparison", loc="left", fontweight="bold",
                 fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=170)
    plt.close(fig)


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = list(data.feature_names)
    paper_indices = [names.index(name) for name in PAPER_FEATURES]

    _, all_confusion, _ = evaluate_split(X, y)
    paper_model, paper_confusion, _ = evaluate_split(X[:, paper_indices], y)
    all_cv = evaluate_cv(X, y)
    paper_cv = evaluate_cv(X[:, paper_indices], y)

    print_results("All 30 features", all_confusion, all_cv)
    print_results("Paper's 3 features", paper_confusion, paper_cv)
    weights = paper_model[-1].coef_[0]
    print("\nThree-feature standardized weights:")
    for name, weight in zip(PAPER_FEATURES, weights):
        print(f"  {name}: {weight:.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    style_axes()
    plot_distributions(X, y, paper_indices)
    plot_confusions(all_confusion, paper_confusion)
    plot_comparison(all_cv, paper_cv)
    print(f"\nSaved figures to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
