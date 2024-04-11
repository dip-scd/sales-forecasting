import pandas as pd
import numpy as np
from typing import Optional

def generate_test_timeseries(length: int = 100, 
                             mean_trend: float = 1., 
                             variance_trend: float = 2,
                             period_freq: float = 3.,
                             noise_std: float = 0.5,
                             start: str = '2000.01.01',
                             freq: str = 'D',
                             random_state: Optional[int] = None
                             ) -> np.ndarray:

    """Generate test timeseries data.

    Args:
        length: Length of the timeseries
        mean_trend: Mean of the trend
        variance_trend: Variance of the trend
        period_freq: Period frequency of the periodic component
        noise_std: Standard deviation of the noise
        start: Start date
        freq: Frequency
        random_state: Random state

    Returns:
        Timeseries data
    """
    arr_trend_mean = np.linspace(0, mean_trend, length)
    arr_periodic = np.sin(np.linspace(0, np.pi*2, length)*period_freq)
    arr_trend_variance = np.linspace(1, variance_trend, arr_trend_mean.shape[0])

    if random_state is not None:
        np.random.seed(random_state)
    arr_noise = np.random.normal(0, noise_std, arr_trend_mean.shape)

    arr_timeseries = arr_trend_variance*(arr_trend_mean + arr_periodic + arr_noise)

    index = pd.date_range(start, periods=length, freq=freq)
    sr_timeseries = pd.Series(arr_timeseries, index=index)
    return sr_timeseries