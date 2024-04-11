import pandas as pd
from IPython.display import display

def transform_forward(
    sr_x: pd.Series, 
    num_rolling: int = 5,
    verbose: bool = False
) -> pd.Series:
    """
    Transforms a time series by differencing and normalizing by rolling standard deviation.

    Parameters:
    sr_x (pd.Series): The input time series
    num_rolling (int): The window size for rolling standard deviation

    Returns:
    pd.Series: The transformed time series
    """

    def _print(str):
        print(str) if verbose else None

    def _display(df):
        display(df) if verbose else None
        

    _print('sr_x')
    _display(sr_x.head(15))

    sr_x_diff = sr_x.diff()
    _print('sr_x_diff')
    _display(sr_x_diff.head(15))

    sr_x_std = sr_x.shift(1).rolling(num_rolling).std()
    _print('sr_x_std')
    _display(sr_x_std.head(15))

    sr_x_transformed = sr_x_diff / sr_x_std
    _print('sr_x_transformed')
    _display(sr_x_transformed.head(15))
    return sr_x_transformed

def transform_reverse(
    sr_x_original_start: pd.Series,
    sr_x_transformed: pd.Series, 
    num_rolling: int = 5,
    verbose: bool = False
) -> pd.Series:
    """
    Transforms a normalized time series back to its original scale.

    Parameters:
    sr_x_original_start (pd.Series): The first `num_rolling` values of the original time series
    sr_x_transformed (pd.Series): The normalized input time series 
    num_rolling (int): The window size for rolling standard deviation

    Returns:
    pd.Series: The restored time series
    """
    def _print(str):
        print(str) if verbose else None

    def _display(df):
        display(df) if verbose else None

    _print('sr_x_transformed')
    _display(sr_x_transformed.head(15))

    _print('sr_x_original_start')
    _display(sr_x_original_start.head(15))

    assert len(sr_x_original_start) >= num_rolling, f'The length of sr_x_original_start ({len(sr_x_original_start)}) must be greater than or equal to num_rolling ({num_rolling})'
    assert sr_x_original_start.shift(1, freq='infer').index[-1] >= sr_x_transformed.index[0], f'Index of sr_x_transformed must start at the next period after the last index of sr_x_original_start (or earlier) ({sr_x_original_start.index[-1]} -x-> {sr_x_transformed.index[0]})'

    if sr_x_original_start.index[-1] >= sr_x_transformed.index[-1]:
        _print('nothing to reverse, returning the sr_x_original_start as is')
        return sr_x_original_start
    
    sr_x_transformed = sr_x_transformed[
        sr_x_transformed.index > sr_x_original_start.index[-1]
    ]

    _print('sr_x_transformed cut')
    _display(sr_x_transformed.head(15))
    
    sr_x_original_std = sr_x_original_start\
        .rolling(num_rolling).std()\
            .shift(1, freq='infer').dropna()
    _print('sr_x_original_std')
    _display(sr_x_original_std.head(15))

    assert len(sr_x_original_std) > 0, \
        f'The length of calclualted rolling std is zero, sr_x_original_start seem to contain not enough values to perform the reverse transformation'

    sr_x_restored_diff = sr_x_transformed * sr_x_original_std
    _print('sr_x_restored_diff')
    _display(sr_x_restored_diff.head(15))

    sr_x_restored_cumulative = sr_x_restored_diff.cumsum()

    _print('sr_x_restored_cumulative')
    _display(sr_x_restored_cumulative.head(15))

    sr_x_restored = sr_x_original_start[sr_x_original_start.index[-1]] + sr_x_restored_cumulative
    sr_x_restored = sr_x_restored.loc[sr_x_original_std.index].dropna()

    sr_x_restored = sr_x_restored.loc[
        [idx for idx in sr_x_restored.index if idx not in sr_x_original_start.index]
    ].dropna()

    _print('sr_x_original_start to concat')
    _display(sr_x_original_start.head(15))
    _print('sr_x_restored to concat')
    _display(sr_x_restored.head(15))

    sr_x_restored = pd.concat([sr_x_original_start, sr_x_restored], axis=0)
    _print('sr_x_restored after concat')
    _display(sr_x_restored.head(15))

    if sr_x_restored.index[-1] < sr_x_transformed.index[-1]:
        _print(f'{sr_x_restored.index[-1]} < {sr_x_transformed.index[-1]}')
        sr_x_restored = transform_reverse(sr_x_restored,
                                          sr_x_transformed, 
                                          num_rolling=num_rolling)

    return sr_x_restored

class TimeseriesTransformerDiffDividedByStd:

    def __init__(self, num_rolling=5):
        self.num_rolling = num_rolling

    def transform_forward(self, sr_x, *args, **kwargs):
        return transform_forward(sr_x, self.num_rolling, *args, **kwargs)
    
    def transform_reverse(self, sr_x_original_start, sr_x_transformed, *args, **kwargs):
        return transform_reverse(sr_x_original_start, sr_x_transformed, self.num_rolling, *args, **kwargs)