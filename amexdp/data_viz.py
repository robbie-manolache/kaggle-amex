
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Data Viz Tools +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def auto_subplots(n_plots: int,
                  n_col: int = 2,
                  multi_row_lim: int = 3,
                  row_dims: tuple = (12, 4)):

    if n_plots <= multi_row_lim:
        plot_dims = (1, n_plots)
    else:
        plot_dims = (int(np.ceil(n_plots/n_col)), n_col)

    r, c = plot_dims
    fig, axes = plt.subplots(r, c, figsize=(row_dims[0], row_dims[1]*r))

    return axes.flatten()


def bar_counter(series: pd.Series,
                bar_type: str = "barh",
                ax: plt.Axes = None,
                figsize: tuple = (12, 4)):

    name = series.name
    counts = series.value_counts().sort_index()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    counts.plot(kind=bar_type, ax=ax)

    for i, n in enumerate(counts):
        ax.text(n, i, str(n), verticalalignment="center")

    ax.set_xlim([0, counts.max()*1.175])

    if name is not None:
        ax.set_ylabel(name)

    if ax is None:
        plt.show()


def signal_preview(df: pd.DataFrame,
                   cols: list,
                   continuous: bool,
                   n_cuts: int = 7,
                   cut_type: str = "quantile",  # "range"
                   plot_dims: tuple or str = "auto",
                   width: int or float = 12,
                   row_height: int or float = 4):

    if plot_dims == "auto":
        if len(cols) <= 3:
            plot_dims = (1, len(cols))
        else:
            plot_dims = (int(np.ceil(len(cols)/2)), 2)

    r, c = plot_dims
    fig, axes = plt.subplots(r, c, figsize=(width*0.8, row_height*r))
    
    if r*c > 1:
        ax_list = axes.flatten()[:len(cols)]
    else:
        ax_list = [axes]

    i = 0
    for a, ax in zip(cols, ax_list):

        if continuous:
            labels = [f"B{i+1}" for i in range(n_cuts)]
            if cut_type == "quantile":
                a = pd.qcut(df[a], q=n_cuts, labels=labels, duplicates="drop")
            else:
                a = pd.cut(df[a], bins=n_cuts, labels=labels, duplicates="drop")

        df.groupby(a)["target"].mean().plot(kind="bar", ax=ax)

        if i == 0 and not continuous:
            ax.set_ylabel("default rate")
        i += 1

    if continuous:
        fig.suptitle(f"{cut_type.capitalize()} Bins")

    fig.tight_layout()
    plt.show()
