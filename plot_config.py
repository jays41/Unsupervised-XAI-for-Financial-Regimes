import matplotlib.pyplot as plt

COLORS = {
    "cluster_0": "#4878CF",
    "cluster_1": "#6ACC65",
    "cluster_2": "#D65F5F",
    "normal":    "#5B9BD5",
    "unusual":   "#ED7D31",
    "positive":  "#4393C3",
    "negative":  "#D6604D",
    "shap":      "#2166AC",
    "random":    "#D6604D",
    "fill":      "#F4A582",
}

CLUSTER_COLORS = [COLORS["cluster_0"], COLORS["cluster_1"], COLORS["cluster_2"]]
BINARY_COLORS  = {"normal": COLORS["normal"], "unusual": COLORS["unusual"]}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi":           300,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "font.family":          "DejaVu Sans",
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.labelsize":       12,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      10,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            False,
        "grid.alpha":           0.3,
        "grid.linestyle":       "--",
        "lines.linewidth":      1.5,
        "lines.markersize":     5,
    })