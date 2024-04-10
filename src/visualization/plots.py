
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.stats as stats
from typing import Tuple, Callable

from src.visualization.html import show_stat_test_results

class S:
    context = '#505050'
    focus = '#F21905'
    neutral = '#039CBF'
    alt = '#204090'

def despine(ax: plt.Axes, full: bool = False, grid: bool = True) -> None:
    """Remove spines from matplotlib axes.

    Args:
        ax : matplotlib.axes.Axes
            The axes to despine
        full : If True, remove all spines and ticks.
        grid : If True, keep gridlines.

    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if full:
        ax.set_xticks([])
        ax.set_yticks([])
    if grid:
        ax.grid()


def make_fig(width: int = 10, height: int = 10, nrows: int = 1,
        ncols: int = 1, squeeze: bool = True, *args, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """Make a matplotlib figure with the given dimensions, squeezing axes as needed.
    
    Args:
        width: Width of the figure in inches
        height: Height of the figure in inches  
        nrows: Number of rows of subplots
        ncols: Number of columns of subplots
        squeeze: Whether to squeeze the returned axes to 1D or 2D if only 1 subplot
        *args: Additional positional arguments to pass to plt.subplots
        **kwargs: Additional keyword arguments to pass to plt.subplots
        
    Returns:
        fig: The matplotlib Figure instance
        ax: The matplotlib Axes instance(s)
    """
    fig, ax = plt.subplots(figsize=(width, height),
                       nrows=nrows, ncols=ncols,
                       facecolor='white',
                        squeeze = squeeze,
                        num=1,clear=True,
                       *args, **kwargs)
    
    if (nrows > 1 and ncols > 1) or not squeeze:
        for axr in ax:
            for axc in axr:
                despine(axc)
    elif nrows == 1 and ncols == 1:
        despine(ax)
    else:
        for axc in ax:
            despine(axc)
    
    return fig, ax


def plot_sales(df_product_sales: pd.DataFrame, 
               show_legend: bool = False, 
               title='Sales per day for differnt product categories',
               fig_size: Tuple[int, int] = (20, 4),) -> None:
    """Plot the sales per day for different product categories.
    
    Args:
        df_product_sales (pd.DataFrame): DataFrame containing the product sales data
        show_legend (bool, optional): Whether to show the legend.
        fig_size (Tuple[int, int], optional): Figure size in inches as (width, height).
        
    Returns:
        None
    """
    
    fig, ax = make_fig(*fig_size)
    df_product_sales.plot(ax=ax, legend=False, alpha=0.5)
    ax.set_ylabel('Sales per day')
    if show_legend:
        ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(None)
    plt.show()

def plot_timeseries(sr_x: pd.Series, ax) -> None:
    """Plot a timeseries with rolling statistics.

    Args:
        sr_x (pd.Series): The timeseries to plot.
        ax (matplotlib.axes.Axes): The axes to plot on.

    """

    sr_x.plot(ax=ax, alpha=1., color=S.neutral)

    num_rolling = 50
    sr_x_mean = sr_x.rolling(num_rolling).mean().dropna()
    sr_x_median = sr_x.rolling(num_rolling).median().dropna()
    sr_x_std = sr_x.rolling(num_rolling).std().dropna()

    alpha=0.5
    sr_x_mean.plot(ax=ax, color=S.focus, alpha=alpha)
    sr_x_median.plot(ax=ax, color=S.context, alpha=alpha, linestyle='--')

    (sr_x_mean + sr_x_std).plot(ax=ax, 
                                color=S.context, 
                                alpha=alpha, linestyle=':')

    (sr_x_mean - sr_x_std).plot(ax=ax, 
                                color=S.context, 
                                alpha=alpha, linestyle=':')

    ax.set_ylim(sr_x.min(), sr_x.max())
    leg = ax.legend([
        'value',
        f'rolling({num_rolling}) mean',
        f'rolling({num_rolling}) median', 
        f'rolling({num_rolling}) 10%, 90%'],
        loc='upper left')
    leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(None)
    ax.set_ylabel('X')
    ax.set_title('Timeseries')

def plot_histogram(
    sr_x: pd.Series, 
    ax: matplotlib.axes.Axes, 
    num_bins: int = 50, 
    hist_range: tuple = None,
    hist_size_ratio: float = None,
    xmin: float = 0, 
    xmax: float = None
) -> None:
    """Plot a histogram of a timeseries with statistics.

    Args:
        sr_x (pd.Series): The timeseries to plot.
        ax (matplotlib.axes.Axes): The axes to plot on.
        num_bins (int): Number of bins for the histogram.
        hist_range (tuple): Min and max values for the histogram range.
        hist_size_ratio (float): Ratio to scale histogram height.
        xmin (float): Minimum value for x-axis.
        xmax (float): Maximum value for x-axis.

    """
    
    def print(str):
        pass
        # print(str)

    if xmax is None:
        xmax = max(sr_x)

    if hist_range is None:
        hist_range = (xmin, xmax)

    standalone = (hist_size_ratio is None)
    density = not standalone
    
    sr_x_slice = sr_x
    slice_len = len(sr_x_slice)

    counts, bins = np.histogram(sr_x_slice, 
                                bins = num_bins,
                                range = hist_range,
                                density = density)
    print(f'len(sr_x_slice) {slice_len}')
    print(f'bins {bins}')
    print(f'counts {counts}')
    
    print(f'xmin xmax  {xmin} {xmax}')

    max_count = max(counts)
    if standalone:
        hist_size_ratio = 1
        counts_scale = 1
        
    counts_scale_base = (slice_len / max_count)
    
    if not standalone:
        counts_scale = counts_scale_base  * hist_size_ratio
        counts = counts * counts_scale
    
    print(f'counts_scale {counts_scale}')
    
    print(f'counts after scaling {counts}')
    
    ax.stairs(xmin+counts, bins, 
               orientation='horizontal', 
               fill=True, 
               color=S.neutral, 
               baseline = xmin
               )
    
    xmax = xmin + (max_count * counts_scale_base  * hist_size_ratio)
    xright = xmin + (max_count * counts_scale_base)
    
    if standalone:
        xmin = 0
        xmax = max_count
        xright = xmin+((xmax-xmin)*1.25)
        ax.set_ylabel('X')
        ax.grid()
        ax.set_title('Timeseries histogram')
        ax.set_xlabel('counts')
        ax.set_xlim(xmin, xright)

        ax.set_xticks(ax.get_xticks()[:-2])
    

    hlines_params = dict(
        xmin=xmin,
        xmax=xmax,
        color=S.context, 
        alpha=0.5,
    )

    print(hlines_params)

    for j, (func_y, name, va, ln_st) in enumerate([
        (lambda sr: pd.Series.quantile(sr, 0.9), '', 'bottom',  ':'),
        (lambda sr: pd.Series.quantile(sr, 0.5), '', 'center', '--'),
        (lambda sr: pd.Series.quantile(sr, 0.1), '', 'top', ':'),
    ]):
        num_y = func_y(sr_x_slice)
        ax.hlines(
            num_y,
            linestyle=ln_st,
            **hlines_params
            )
        ax.text(
            hlines_params['xmax'],
            num_y,
            s=f'   {name} {np.round(num_y, 2)}  ',
            color=S.context,
            horizontalalignment='left',
            verticalalignment=va,
            rotation=0,
            )
        
    num_std = sr_x_slice.std()
    num_mean = sr_x_slice.mean()

    ax.text(
            xright,
            hist_range[1],
            s=f'\nmean: {np.round(num_mean,2)}  \n  std dev: {np.round(num_std,2)}  ',
            color=S.context,
            horizontalalignment='right',
            verticalalignment='top',
            rotation=0,
            )

    if hist_range is not None:
        ax.set_ylim(hist_range)

    legend_elements = [
        Line2D([0], [0], color=S.context, lw=1, label='90%', linestyle=':'),
        Line2D([0], [0], color=S.context, lw=1, label='median', linestyle='--'),
        Line2D([0], [0], color=S.context, lw=1, label='10%', linestyle=':'),
    ]
   
    if standalone and False:
        leg = ax.legend(handles=legend_elements, 
                        loc='upper right',
                        bbox_to_anchor=(1.08, 0.8),)
        
        for text in leg.get_texts():
            text.set_color(S.context)
        leg.get_frame().set_linewidth(0.0)


def plot_histogram_series(sr_x: pd.Series, ax: plt.Axes, num_hists: int = 5) -> list[pd.Timestamp]:
    """Plot histograms of slices of a timeseries.

    Args:
        sr_x: Timeseries to slice and plot histograms.
        ax: Axes to plot on.
        num_hists: Number of histogram slices.

    Returns:
        lst_xticks: List of x-axis tick locations.
    """

    def print(str):
        pass
        # print(str)

    slice_len = len(sr_x) // num_hists

    ax.set_xlim(sr_x.index[0], sr_x.index[-1])

    lst_xticks = [
            sr_x.index[i*slice_len] for i in range(num_hists)
        ]
    for xtick in lst_xticks:
        ax.axvline(xtick, color='black', alpha=1., linewidth=0.2)

    ax.set_ylabel('X')
    ax.grid()

    ax.set_title('Histograms of timeseries slices')

    ax_twin = ax.twiny()
    ax_twin.set_xlim(0, len(sr_x))
    ax_twin.grid(False)
    ax_twin.set_xticks([])
    despine(ax_twin)

    x_min = sr_x.min()
    x_max = sr_x.max()
    
    hist_size_ratio = .6
    print(f'slice_len {slice_len}')
    print(f'hist_size_ratio {hist_size_ratio}')
    for i in range(num_hists):
        sr_x_slice = sr_x[(i)*slice_len:(i+1)*slice_len]
        imin = i * len(sr_x_slice)
        imax = imin + len(sr_x_slice)
        print(f'imin imax {imin} {imax}' )

        plot_histogram(sr_x_slice, ax_twin, num_bins=50, 
                       hist_range = (x_min, x_max),
                       hist_size_ratio=hist_size_ratio,
                       xmin=imin, xmax=imax)

    return lst_xticks


def calc_sr_autocorr(sr: pd.Series, max_lag: int = 100) -> pd.Series:
    """Calculate autocorrelation for a pandas Series up to max_lag.

    Args:
        sr: Input pandas Series
        max_lag: Maximum lag to calculate autocorrelation

    Returns: 
        pd.Series containing autocorrelation values for lags up to max_lag
    """
    dict_ret = {}
    for l in range(1, max_lag+1):
        dict_ret[l] = sr.autocorr(lag=l)
    return pd.Series(dict_ret)

def plot_autocorrelation(sr_x_slice: pd.Series, ax: plt.Axes) -> None:
    """Plot the autocorrelation of a pandas Series.

    Args:
        sr_x_slice: The input pandas Series to calculate autocorrelation.
        ax: The matplotlib Axes to plot the autocorrelation on.

    Returns:
        None
    """
    calc_sr_autocorr(sr_x_slice, 45).plot.bar(
        ax=ax, color=S.neutral, width=0.5, alpha=1)
    ax.set_xticks(range(-1,45, 10))
    ax.set_ylabel('correlation')
    ax.set_xlabel('lag')
    ax.set_title('Autocorrelation')

def plot_rolling_stat(sr_x: pd.Series, ax: plt.Axes, func_stat: Callable, ylabel: str) -> None:
    """Plot a rolling statistic of a pandas Series.
    
    Args:
        sr_x: Input pandas Series.
        ax: Matplotlib Axes to plot on.
        func_stat: Function to calculate statistic on rolling window.
        ylabel: Label for y-axis.
        
    Returns:
        None
    """
    for rolling_period in range(10,500, 10):
        sr_x.rolling(rolling_period).apply(func_stat).plot(ax=ax, color=S.neutral, alpha=0.1)
    ax.set_ylabel(ylabel)
    
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.margins(0.)
    ax.grid()

def plot_normal_probplot(sr_x: pd.Series, ax: plt.Axes) -> None:
    """Generate a normal probability plot for a pandas Series.

    Args:
        sr_x: Input pandas Series
        ax: Matplotlib Axes to plot on

    Returns:
        None
    """
    stats.probplot(sr_x, dist="norm", plot=ax)

    ax.set_title("Normal Probability Plot")
    line0 = ax.get_lines()[0]
    line0.set_marker('.') 
    line0.set_markerfacecolor(S.neutral)
    line0.set_markeredgewidth(0)

    line1 = ax.get_lines()[1]
    line1.set_color(S.context)

def show_timeseries_plots(sr_x: pd.Series) -> None:
    """Show various timeseries plots for a pandas Series.
    
    Args:
        sr_x: Input pandas Series
        
    Returns:
        None
    """
    sr_x = sr_x.dropna()
    x_min, x_max = min(sr_x), max(sr_x)

    fig = plt.figure(layout="constrained", figsize=(15,10))

    gs = GridSpec(4,2, figure=fig, width_ratios=[1, 0.25], 
                        height_ratios=[1, 1, 0.5, 0.5])

    ax_0_0 = fig.add_subplot(gs[0, 0])
    ax_0_1 = fig.add_subplot(gs[0, 1])
    ax_1_0 = fig.add_subplot(gs[1, 0])
    ax_1_1 = fig.add_subplot(gs[1, 1])
    ax_2_0 = fig.add_subplot(gs[2, 0])
    ax_3_0 = fig.add_subplot(gs[3, 0])

    ax_23_1 = fig.add_subplot(gs[2:4, 1])

    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            despine(ax)

    format_axes(fig)

    ax = ax_1_0
    lst_xticks = plot_histogram_series(sr_x, ax, num_hists = 6)
    ax.set_xticklabels([])

    ax = ax_1_1
    plot_histogram(sr_x, ax, num_bins=50, hist_range=(x_min, x_max))

    ax = ax_0_0
    plot_timeseries(sr_x, ax)
    for xtick in lst_xticks:
        ax.axvline(xtick, color='black', alpha=1., linewidth=0.2)

    ax=ax_0_1
    plot_autocorrelation(sr_x, ax)


    ax = ax_2_0
    plot_rolling_stat(sr_x, ax, np.mean, 'rolling\nmean')

    ax = ax_3_0
    plot_rolling_stat(sr_x, ax, np.std, 'rolling\nstd')

    ax = ax_23_1
    plot_normal_probplot(sr_x, ax)

    plt.show()

def show_timeseries_analysis(sr_x: pd.Series) -> None:
    """Display timeseries analysis plots and statistical test results.
    
    Args:
        sr_x: Time series data
        
    Returns:
        None
    """
    
    show_timeseries_plots(sr_x)
    show_stat_test_results(sr_x)

def show_comparison_original_vs_restored(sr_x_original: pd.Series, timeseries_transformer) -> None:

    sr_x_transformed = timeseries_transformer.transform_forward(sr_x_original)
    sr_x_restored = timeseries_transformer.transform_reverse(
        sr_x_transformed, sr_x_original[:timeseries_transformer.num_rolling])

    fig, axx = plt.subplots(figsize=(15, 6), nrows=2, gridspec_kw={'height_ratios': [3, 1]})
    ax=axx[0]
    despine(ax)
    sr_x_original.plot(color=S.neutral, alpha=0.5, ax=ax)
    sr_x_restored.plot(color=S.focus, linestyle='--', alpha=0.5, ax=ax)
    ax.set_title('Original vs Restored time series')
    ax.set_xlabel(None)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    leg = ax.legend(['Original', 'Restored'])
    leg.get_frame().set_linewidth(0.0)

    ax=axx[1]
    despine(ax)
    sr_x_restoration_delta = sr_x_restored - sr_x_original
    sr_x_restoration_delta.plot(color=S.context, ax=ax)
    ax.set_ylim(-1, 1)
    ax.set_title('Difference between the original and restored time series')
    ax.set_xlabel(None)
    
    fig.tight_layout()
    
    plt.show()