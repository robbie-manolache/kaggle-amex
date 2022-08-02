
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+
# Data Viz Tools +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-
# +-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+(*)+-|.|-+


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bar_counter(series: pd.Series,
                ax: plt.Axes = None,
                figsize: tuple = (12, 4)):

    name = series.name
    counts = series.value_counts().sort_index()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    counts.plot(kind="barh", ax=ax)

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
                   n_cuts: int = None,
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
    fig, axes = plt.subplots(r, c, figsize=(width, row_height*r))

    i = 0
    for a, ax in zip(cols, axes.flatten()[:len(cols)]):

        if continuous:
            if n_cuts is None:
                n_cuts = 7
            if cut_type == "quantile":
                a = pd.cut(df[a].round(4), bins=n_cuts)
            else:
                a = pd.qcut(df[a].round(4), q=n_cuts)

        df.groupby(a)["target"].mean().plot(kind="bar", ax=ax)

        if i == 0:
            ax.set_ylabel("default rate")
        i += 1

    plt.show()
