
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import scipy
import scipy.stats as stats
from itertools import product
import seaborn as sns
from typing import Callable, Tuple, Any, Dict, List

from src.features.make_model_dataset import get_x_y
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

def plot_points_density(sr: pd.Series, 
                        ax: plt.Axes = None,
                        title: str = None) -> None:
    """
    Plot the availability of data over time.

    Args:
        sr (pd.Series): A pandas Series containing the data availability information.
        ax (plt.Axes, optional): An existing Axes object to plot on. If not provided, a new figure and axes will be created.
        title (str, optional): The title of the plot.
    Returns:
        None
    """
    standalone = False
    if ax is None:
        standalone = True
        _, ax = make_fig(15, 1.3)
    sr.plot(marker='|', linewidth=0., color='#00000010', ax=ax)
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.grid(axis='x')

    if title is not None:
        ax.set_title(title)

    if standalone:
        plt.show()

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

def plot_dist_probplot(
        sr_x: pd.Series,
        ax: plt.Axes,
        dist: Any = "norm",
        dist_name: str = "Normal",
        *args, **kwargs) -> None:
    """Generate a distribution probability plot for a pandas Series.

    Args:
        sr_x: Input pandas Series
        ax: Matplotlib Axes to plot on

    Returns:
        None
    """
    stats.probplot(sr_x, dist=dist, plot=ax, *args, **kwargs)

    ax.set_title(f"{dist_name} Probability Plot")
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

    gs = GridSpec(3,5, figure=fig,)

    ax_0_0 = fig.add_subplot(gs[0, 0:4])
    ax_0_1 = fig.add_subplot(gs[0, 4])
    ax_1_0 = fig.add_subplot(gs[1, 0:4])
    ax_1_1 = fig.add_subplot(gs[1, 4])
    ax_2_0 = fig.add_subplot(gs[2, 0])
    ax_2_1 = fig.add_subplot(gs[2, 1])
    ax_2_2 = fig.add_subplot(gs[2, 2])
    # ax_2_3 = fig.add_subplot(gs[2, 3])
    # ax_2_4 = fig.add_subplot(gs[2, 4])

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


    # ax = ax_2_0
    # plot_rolling_stat(sr_x, ax, np.mean, 'rolling\nmean')

    # ax = ax_3_0
    # plot_rolling_stat(sr_x, ax, np.std, 'rolling\nstd')

    ax = ax_2_0
    plot_dist_probplot(sr_x, ax, 'norm', 'Normal')

    ax = ax_2_1
    plot_dist_probplot(sr_x, ax, scipy.stats.powerlaw(1), 'powerlaw(1)')

    ax = ax_2_2
    plot_dist_probplot(sr_x, ax, scipy.stats.gamma(1), 'gamma(1)')

    # ax = ax_2_3
    # plot_dist_probplot(sr_x, ax, scipy.stats.expon, 'expon')

    # ax = ax_2_4
    # plot_dist_probplot(sr_x, ax, scipy.stats.exponnorm(1), 'exponnorm')

    

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
        sr_x_original[:timeseries_transformer.num_rolling], sr_x_transformed)

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

def plot_series_errors(
        lst_series_specs: List[Tuple[pd.Series, str, Dict[str, Any]]],
        lst_errors: List[float],
        ax: plt.Axes,
        metric_name: str,
        dataset_name: str
        ) -> None:
    """
    Plots the errors for a list of series specifications on a given axis.

    Args:
        lst_series_specs (List[Tuple[pd.Series, str, Dict[str, Any]]]): A list of tuples containing the series data, label, and plot style dictionary.
        lst_errors (List[float]): A list of error values corresponding to each series specification.
        ax (plt.Axes): The axis object on which to plot the errors.
        dataset_name (str): The name of the dataset being plotted.

    Returns:
        None
    """
    
    dict_val_error: Dict[str, float] = {}

    for spec, error in zip(lst_series_specs, lst_errors):
        dict_val_error[spec[1]] = error

    df_val_error = pd.Series(dict_val_error).rename('Error').to_frame()

    df_val_error['Error'].plot.bar(
        ax=ax, 
        color=[
            e[2]['color'] for e in lst_series_specs
        ],
        width=0.5, alpha=1.
    )
    ax.bar_label(ax.containers[0], padding=3, fmt='%.3f')

    ax.grid(axis='y')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}\n on the {dataset_name} set')


def plot_model_predictions_transform_reversed(
        lst_series_specs: List[Tuple[pd.Series, str, Dict]],
        ax: plt.Axes,
        dataset_name: str
    ) -> None:
    """
    Plot the model predictions on the given axis.

    Args:
        lst_series_specs (List[Tuple[pd.Series, str, Dict]]): A list of tuples containing the series data, label, and plot options.
        ax (plt.Axes): The axis object on which to plot the data.
        dataset_name (str): The name of the dataset being plotted.

    Returns:
        None
    """

    for series_spec in lst_series_specs:
        series_spec[0].plot(ax=ax, **series_spec[2])

    leg = ax.legend([
            e[1] for e in lst_series_specs
        ])
    leg.get_frame().set_linewidth(0.0)
    ax.set_title(f'Model forecast on the {dataset_name} set')


def show_forecast_errors_and_predictions(
    df: pd.DataFrame,
    model: Any,
    error_metric: Tuple[Callable, str],
    dataset_name: str
) -> None:
    """
    Plots the forecast errors and predictions for a given model and dataset.

    Args:
        df (pd.DataFrame): DataFrame with features and labels..
        model (Any): The model object used for forecasting.
        error_metric (Tuple[Callable, str]): A tuple containing the error metric function and its name.
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    df_X_val, df_y_val = get_x_y(df)

    # Real values i.e ideal predictions
    sr_y_val = df[('y', 1)]

    sr_val_baseline_naive = df[('baseline', 'naive')]
    sr_val_baseline_linreg = df[('baseline', 'linear_regression')]


    yhat_val = model.predict(df_X_val)
    sr_yhat_val = pd.Series(
        yhat_val,
        index=df_X_val.index
    )

    lst_series_specs = [
        (sr_y_val, 'Real values i.e. ideal forecast', dict(color=S.context, alpha=0.5)),
        (sr_val_baseline_naive, 'Naive baseline', dict(color=S.alt, linestyle=':', alpha=0.5)),
        (sr_val_baseline_linreg, 'Linear regression\n(done on transformed values)\nbaseline', dict(color=S.neutral, linestyle='-', alpha=0.5)),
        (sr_yhat_val, 'Model forecast', dict(color=S.focus, alpha=1.)),
    ]

    lst_errors = [
        error_metric[0](
            df_y_val, 
            e[0].loc[
                sr_yhat_val.index
            ]
            ) for e in lst_series_specs
    ]

    fig, axx = make_fig(15, 9, ncols=2, gridspec_kw={'width_ratios': [1, 3]})

    plot_model_predictions_transform_reversed(
        lst_series_specs,
        ax = axx[1],
        dataset_name = dataset_name
    )

    plot_series_errors(
        lst_series_specs[1:],
        lst_errors[1:],
        ax = axx[0],
        metric_name = error_metric[1],
        dataset_name = dataset_name
    )

    fig.tight_layout()
    plt.show()



def show_forecast_errors_and_predictions_transform_reversed(
    df: pd.DataFrame,
    df_original: pd.DataFrame,
    model: Any,
    error_metric: Tuple[Callable,str],
    timeseries_transformer,
    dataset_name: str
) -> None:
    """
    Displays forecast errors and predictions with the transformed values reversed back to the original scale.

    Args:
        df (pd.DataFrame): DataFrame with the transformed features and target values.
        df_original (pd.DataFrame): DataFrame with features and target values in the original scale.
        model (Any): The trained model to make predictions.
        error_metric (Tuple[Callable, str]): A tuple containing the error metric function and its name.
        timeseries_transformer: The time series transformer used to transform the data.
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    df_X_val, _ = get_x_y(df)

    sr_x_val_original = df_original[('x','-0')]

    # Real values i.e ideal predictions
    sr_y_val_original = df_original[('y', 1)]

    # baseline linear regression that uses original values as an input
    sr_val_original_baseline_linreg = df_original[('baseline', 'linear_regression')]

    # baseline naive that uses transformed values as an input
    # excepted to be equal to sr_val_original_baseline_naive
    sr_val_transformed_baseline_naive_reversed = timeseries_transformer.transform_forecast_reverse(
        sr_x_val_original, df[('baseline', 'naive')])

    # baseline linear regression that uses transformed values as an input
    sr_val_transformed_baseline_linreg_reversed = timeseries_transformer.transform_forecast_reverse(
        sr_x_val_original, df[('baseline', 'linear_regression')])


    yhat_val = model.predict(df_X_val)
    sr_yhat_val = pd.Series(
        yhat_val,
        index=df_X_val.index
    )

    sr_val_forecast_reversed = timeseries_transformer.transform_forecast_reverse(
        sr_x_val_original, sr_yhat_val)

    df_val_original_cut = df_original.loc[
        sr_val_forecast_reversed.index
    ]

    _, df_y_val_original_cut = get_x_y(df_val_original_cut)

    lst_series_specs = [
        (sr_y_val_original, 'Real values i.e. ideal forecast', dict(color=S.context, alpha=0.5)),
        (sr_val_transformed_baseline_naive_reversed, 'Naive baseline', dict(color=S.alt, linestyle=':', alpha=0.5)),
        (sr_val_original_baseline_linreg, 'Linear regression\n(done directly on original values)\nbaseline', dict(color=S.neutral, linestyle='--', alpha=0.5)),
        (sr_val_transformed_baseline_linreg_reversed, 'Linear regression\n(done on transformed values)\nbaseline', dict(color=S.neutral, linestyle='-', alpha=0.5)),
        (sr_val_forecast_reversed, 'Model forecast', dict(color=S.focus, alpha=1.)),
    ]

    lst_errors = [
        error_metric[0](
            df_y_val_original_cut, 
            e[0].loc[
                sr_val_forecast_reversed.index
            ]
            ) for e in lst_series_specs
    ]

    fig, axx = make_fig(15, 9, ncols=2, gridspec_kw={'width_ratios': [1, 3]})

    plot_model_predictions_transform_reversed(
        lst_series_specs,
        ax = axx[1],
        dataset_name = dataset_name
    )

    plot_series_errors(
        lst_series_specs[1:],
        lst_errors[1:],
        ax = axx[0],
        metric_name = error_metric[1],
        dataset_name = dataset_name
    )

    fig.tight_layout()
    plt.show()

def show_forecast_errors_and_predictions_transform_reversed(
    df_val: pd.DataFrame,
    df_val_original: pd.DataFrame,
    model: Any,
    error_metric: Tuple[Callable,str],
    timeseries_transformer,
    dataset_name: str
) -> None:
    
    df_X_val, _ = get_x_y(df_val)

    sr_x_val_original = df_val_original[('x','-0')]

    # Real values i.e ideal predictions
    sr_y_val_original = df_val_original[('y', 1)]

    # baseline linear regression that uses original values as an input
    sr_val_original_baseline_linreg = df_val_original[('baseline', 'linear_regression')]

    # baseline naive that uses transformed values as an input
    # excepted to be equal to sr_val_original_baseline_naive
    sr_val_transformed_baseline_naive_reversed = timeseries_transformer.transform_forecast_reverse(
        sr_x_val_original, df_val[('baseline', 'naive')])

    # baseline linear regression that uses transformed values as an input
    sr_val_transformed_baseline_linreg_reversed = timeseries_transformer.transform_forecast_reverse(
        sr_x_val_original, df_val[('baseline', 'linear_regression')])


    yhat_val = model.predict(df_X_val)
    sr_yhat_val = pd.Series(
        yhat_val,
        index=df_X_val.index
    )

    sr_val_forecast_reversed = timeseries_transformer.transform_forecast_reverse(
        sr_x_val_original, sr_yhat_val)

    df_val_original_cut = df_val_original.loc[
        sr_val_forecast_reversed.index
    ]

    _, df_y_val_original_cut = get_x_y(df_val_original_cut)

    lst_series_specs = [
        (sr_y_val_original, 'Real values i.e. ideal forecast', dict(color=S.context, alpha=0.5)),
        (sr_val_transformed_baseline_naive_reversed, 'Naive baseline', dict(color=S.alt, linestyle=':', alpha=0.5)),
        (sr_val_original_baseline_linreg, 'Linear regression\n(done directly on original values)\nbaseline', dict(color=S.neutral, linestyle='--', alpha=0.5)),
        (sr_val_transformed_baseline_linreg_reversed, 'Linear regression\n(done on transformed values)\nbaseline', dict(color=S.neutral, linestyle='-', alpha=0.5)),
        (sr_val_forecast_reversed, 'Model forecast', dict(color=S.focus, alpha=1.)),
    ]

    lst_errors = [
        error_metric[0](
            df_y_val_original_cut, 
            e[0].loc[
                sr_val_forecast_reversed.index
            ]
            ) for e in lst_series_specs
    ]

    fig, axx = make_fig(15, 9, ncols=2, gridspec_kw={'width_ratios': [1, 3]})

    plot_model_predictions_transform_reversed(
        lst_series_specs,
        ax = axx[1],
        dataset_name = dataset_name
    )

    plot_series_errors(
        lst_series_specs[1:],
        lst_errors[1:],
        ax = axx[0],
        metric_name = error_metric[1],
        dataset_name = dataset_name
    )

    fig.tight_layout()
    plt.show()


def plot_ax_error_grid(ax, 
                       levels: List[float], 
                       x: List[float], y: List[float], z: List[float], 
                       x_spec: List[float], y_spec: List[float], 
                       str_x_dimension: str, str_y_dimension: str,
                       title: str):
    """
    Plots a grid of error values on the given axis.

    Args:
        ax: The axis object to plot on.
        levels (List[float]): The contour levels to use for the plot.
        x (List[float]): The x-coordinates of the grid points.
        y (List[float]): The y-coordinates of the grid points.
        z (List[float]): The error values at each grid point.
        x_spec (List[float]): The x-tick locations.
        y_spec (List[float]): The y-tick locations.
        str_x_dimension (str): The name of the x-dimension.
        str_y_dimension (str): The name of the y-dimension.
        title (str): The title of the plot.

    Returns:
        The contour plot object.
    """
    cmap = sns.color_palette("magma", as_cmap=True)

    cs = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)
    ax.set_xlabel(str_x_dimension)
    ax.set_ylabel(str_y_dimension)

    min_x = None
    min_y = None
    color_base = S.neutral
    for i, z_item in enumerate(z):
        size = 8
    
        color = f'{color_base}ff'
        # color = f'{S.context}90'
        bbox_props = None
        if z_item <= z.min():
            
            color = '#00DDFF'
            back_color = '#000000'
            bbox_props = dict(boxstyle="Square", fc=back_color, ec=back_color, alpha=0.9)
            size = 12

            min_x = x[i]
            min_y = y[i]

        ax.text(x[i], y[i], np.round(z_item,3), 
                bbox=bbox_props,
                ha='center', va='center', fontdict={
                    'size': size, 
                    'color': color,
                    })

    ax.set_xticks(x_spec)
    ax.set_yticks(y_spec)

    x_padding = (x_spec[-1]-x_spec[0])*0.1
    y_padding = (y_spec[-1]-y_spec[0])*0.1
    ax.set_xlim(x_spec[0]-x_padding, x_spec[-1]+x_padding)
    ax.set_ylim(y_spec[0]-x_padding, y_spec[-1]+x_padding)

    ax.set_title(title)
    ax.grid(True, linestyle=':', color='#00000020', alpha=0.2)

    return cs

def plot_error_grid(x: List[float], y: List[float], z_train: List[float], 
                    z_val: List[float], x_spec: List[float], y_spec: List[float], 
                    row_name: str, 
                    str_x_dimension: str,
                    str_y_dimension: str,
                    levels: List[float], axx: list, fig: plt.Figure) -> None:
    """
    Plots the error grid for both train and validation data.

    Args:
        x (List[float]): The list of x values.
        y (List[float]): The list of y values.
        z_train (List[float]): The list of training error values.
        z_val (List[float]): The list of validation error values.
        x_spec (List[float]): The list of x-axis tick values.
        y_spec (List[float]): The list of y-axis tick values.
        row_name (str): The name of the row.
        str_x_dimension (str): The name of the x-axis dimension.
        str_y_dimension (str): The name of the y-axis dimension.
        levels (List[float]): The list of contour levels.
        axx (list): The list of axes objects.
        fig (plt.Figure): The figure object.

    Returns:
        None
    """
    cs = plot_ax_error_grid(axx[0], levels, x, y, z_train, 
                            x_spec, y_spec, 
                            str_x_dimension, str_y_dimension,
                            f'{row_name}\nTrain error')
    cs = plot_ax_error_grid(axx[1], levels, x, y, z_val, 
                            x_spec, y_spec,
                            str_x_dimension, str_y_dimension,
                              f'{row_name}\nValidation error')

    ax = axx[2]
    ax.set_axis_off()
    
    cbar = fig.colorbar(cs, drawedges=False, 
                        aspect = 50, label='Error', ax=ax,
                        fraction=1.5)
    cbar.outline.set_visible(False)


def show_model_errors_grid(
        lst_grid_params: list, 
        lst_grid_models: list,
        param_grid_definition: dict,
        str_x_dimension: str,
        str_y_dimension: str) -> None:
    """
    Plots the error grids for a grid search of hyperparameters.

    Args:
        lst_grid_params (list): A list of dictionaries containing the hyperparameters for each model.
        lst_grid_models (list): A list of dictionaries containing the trained models and their errors.
        param_grid_definition (dict): A dictionary defining the hyperparameter grid.
        str_x_dimension (str): The name of the hyperparameter to use for the x-axis.
        str_y_dimension (str): The name of the hyperparameter to use for the y-axis.

    Returns:
        None
    """
    dict_rest_dimensions = {
        k:v for k,v in param_grid_definition.items() if k not in [str_x_dimension, str_y_dimension]
    }

    num_rest_dimensions = len(dict_rest_dimensions.keys())

    lst_rest_dimensions = [
            dict(
                zip(dict_rest_dimensions.keys(), values)) \
                      for values in product(*dict_rest_dimensions.values())
        ]
    
    def passes_filter(dict_params, dict_rest_dimension):
        if not all([dict_params[k] == dict_rest_dimension[k] for k in dict_rest_dimension]):
            return False
        return True

    dict_restdim_lst_grid_params_filtered = {}
    for restdim in lst_rest_dimensions:
        lst_grid_params_filtered = []
        lst_grid_models_filtered = []
        for i, r in enumerate(lst_grid_params):
            if passes_filter(r, restdim):
                lst_grid_params_filtered.append(lst_grid_params[i])
                lst_grid_models_filtered.append(lst_grid_models[i])

        str_key = ',\n'.join([
                f'{k} = {restdim[k]}' for k in restdim.keys()
            ])
        dict_restdim_lst_grid_params_filtered[str_key] = {
            'lst_grid_params_filtered': lst_grid_params_filtered,
            'lst_grid_models_filtered':lst_grid_models_filtered
        }

    nrows = len(dict_restdim_lst_grid_params_filtered.keys())
    fig, axxx = make_fig(20,8*nrows, 
                        ncols=3,
                        nrows=nrows,
                        gridspec_kw={
                            'width_ratios': [1,1,0.2],
                            'wspace':0.1,
                            'hspace': 0.1+(0.1*num_rest_dimensions)
                            },
                        squeeze=False
                    )

    def plot_error_grid_row(
            lst_grid_params, 
            lst_grid_models,
            row_name,
            levels,
            axx,
            fig):
        x = np.array([
            e[str_x_dimension] for e in lst_grid_params
        ])

        x_spec = param_grid_definition[str_x_dimension]

        y = np.array([
            e[str_y_dimension] for e in lst_grid_params
        ])

        y_spec = param_grid_definition[str_y_dimension]

        z_train = np.array([
            e['train_error'] for e in lst_grid_models
        ])
        z_val = np.array([
            e['val_error'] for e in lst_grid_models
        ])

        plot_error_grid(x, y, z_train, z_val, x_spec, y_spec, row_name, 
                        str_x_dimension=str_x_dimension,
                        str_y_dimension=str_y_dimension,
                        levels = levels,
                        axx=axx, fig=fig)


    zmax = np.array([e['train_error'] for e in  lst_grid_models]+\
             [e['val_error'] for e in  lst_grid_models])
    zmax = np.quantile(zmax, 0.99, )
    zpadding = 0.1 * zmax
    levels = np.linspace(0, zmax+zpadding, 160)

    for i, restdim in enumerate(dict_restdim_lst_grid_params_filtered):
        lst_grid_params_filtered = dict_restdim_lst_grid_params_filtered[restdim]['lst_grid_params_filtered']
        lst_grid_models_filtered = dict_restdim_lst_grid_params_filtered[restdim]['lst_grid_models_filtered']
        plot_error_grid_row(
            lst_grid_params_filtered, 
            lst_grid_models_filtered, 
            row_name=restdim,
            levels = levels,
            axx=axxx[i], 
            fig=fig)
        
    plt.show()

def plot_error_distirbution_per_hyperparam(df_models: pd.DataFrame,
                                           param_grid_definition, 
                                           col_hyperparam: tuple, 
                                           ax: plt.Axes,
                                           draw_scatter: bool = False) -> None:
    """
    Plots the distribution of training and validation errors for a given hyperparameter.

    Args:
        df_models (pd.DataFrame): DataFrame containing the model results.
        col_hyperparam (tuple): Tuple containing the column name for the hyperparameter.
        ax (plt.Axes): Axes object to plot on.
        draw_scatter (bool, optional): Flag to draw scatter plot. Defaults to False.

    Returns:
        None
    """
    col_error_train = ('error', 'train_error')
    col_error_val = ('error', 'val_error')
    
    if draw_scatter:
        df_models.plot(
            y=col_error_val,
            x=col_hyperparam,
            ax=ax,
            kind='scatter',
            s=20,
            marker='o',
            linewidth=0,
            alpha=.1,
            color=S.context,
        )
    
    df_models.groupby(
        col_hyperparam
        ).agg({col_error_val:'mean'})[
            col_error_val
        ].plot(ax=ax, 
            alpha=0.8,
            color=S.neutral)
    
    df_models.groupby(
        col_hyperparam
        ).agg({col_error_train:'mean'})[
            col_error_train
        ].plot(ax=ax, 
            alpha=0.5,
            color=S.focus)
    
    leg = ax.legend(['Validation error', 'Training error'])
    leg.get_frame().set_linewidth(0.0)

    ax.set_xticks(param_grid_definition[col_hyperparam[1]])
    ax.set_xlabel(col_hyperparam[1])
    ax.set_ylabel('Average error')

def show_error_distirbutions_per_hyperparam(param_grid_definition: dict, df_models: pd.DataFrame) -> None:
    """
    Plots the distribution of training and validation errors for each hyperparameter in the param_grid_definition.

    Args:
        param_grid_definition (dict): A dictionary containing the hyperparameters and their values.
        df_models (pd.DataFrame): A DataFrame containing the model results.

    Returns:
        None
    """

    max_cols = 2
    n_params = len(param_grid_definition)

    num_cols = min(max_cols, n_params)
    num_rows = int(math.ceil(n_params / num_cols))

    fig, axxx = make_fig(6*num_cols, 3*num_rows,
                        ncols=num_cols, 
                        nrows=num_rows,
                        squeeze=False,
                        )

    for i in range(num_rows):
        for j in range(num_cols):
            idx = i*num_cols + j
            if idx >= n_params:
                #hide axes
                axxx[i,j].set_axis_off()
            else:
                col_hyperparam = ('param', list(param_grid_definition.keys())[idx])            
                ax = axxx[i][j]

                plot_error_distirbution_per_hyperparam(
                    df_models, param_grid_definition, col_hyperparam, ax)
        fig.tight_layout()