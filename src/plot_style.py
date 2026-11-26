"""
plot_style.py
=============
One consistent, colour-blind-friendly style for every figure in the project.
Blue and orange are distinguishable for the common types of colour blindness.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"      # main colour / class 'no'
ORANGE = "#eb6834"    # highlight colour / class 'yes'
GREY = "#52514e"      # reference lines and secondary text
CLASS_COLORS = {"no": BLUE, "yes": ORANGE}

# Diverging map for correlations: blue (negative) - light grey (0) - orange (positive)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_grey_orange", [BLUE, "#f0efec", ORANGE])


def apply_style():
    """Call once at the top of every notebook."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e6e6e3",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })
