"""One quiet plotting style for every figure (matches the web page: ink, one blue, one orange)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, BLUE, ORANGE, GREEN, GRAY, RULE = "#1b1b1a", "#1f4fa3", "#c2561b", "#2c6e49", "#8a8a85", "#d9d6cf"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7", "savefig.facecolor": "#fbfaf7",
            "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
            "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
            "figure.dpi": 110, "savefig.dpi": 150,
        }
    )  # fmt: skip
