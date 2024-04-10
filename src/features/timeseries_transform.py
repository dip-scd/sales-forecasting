import pandas as pd

class TimeseriesTransformerDiffDividedByStd:

    @classmethod
    def _transform_forward(
        cls, 
        sr_x: pd.Series, 
        num_rolling: int = 5
    ) -> pd.Series:
        """
        Transforms a time series by differencing and normalizing by rolling standard deviation.

        Parameters:
        sr_x (pd.Series): The input time series
        num_rolling (int): The window size for rolling standard deviation

        Returns:
        pd.Series: The transformed time series
        """

        def print(str):
            pass

        def display(df):
            pass

        sr_x_diff = sr_x.diff()
        print('sr_x_diff')
        display(sr_x_diff.head(15))

        sr_x_std = sr_x.shift(1).rolling(num_rolling).std()
        print('sr_x_std')
        display(sr_x_std.head(15))

        sr_x_transformed = sr_x_diff / sr_x_std
        print('sr_x_transformed')
        display(sr_x_transformed.head(15))
        return sr_x_transformed

    @classmethod
    def _transform_reverse(
        cls,
        sr_x_transformed: pd.Series, 
        sr_x_original_start: pd.Series,
        num_rolling: int = 5
    ) -> pd.Series:
        """
        Transforms a normalized time series back to its original scale.

        Parameters:
        sr_x_transformed (pd.Series): The normalized input time series 
        sr_x_original_start (pd.Series): The first `num_rolling` values of the original time series
        num_rolling (int): The window size for rolling standard deviation

        Returns:
        pd.Series: The restored time series
        """
        def print(str):
            pass

        def display(df):
            pass

        print('sr_x_transformed')
        display(sr_x_transformed.head(15))

        print('sr_x_original_start')
        display(sr_x_original_start.head(15))
        
        sr_x_original_std = sr_x_original_start.rolling(num_rolling).std().shift(1, freq='infer')
        print('sr_x_original_std')
        display(sr_x_original_std.head(15))

        sr_x_restored_diff = sr_x_transformed * sr_x_original_std
        print('sr_x_restored_diff')
        display(sr_x_restored_diff.head(15))

        sr_x_restored_cumulative = sr_x_restored_diff.cumsum()

        print('sr_x_restored_cumulative')
        display(sr_x_restored_cumulative.head(15))

        sr_x_restored = sr_x_original_start.iloc[num_rolling-1] + sr_x_restored_cumulative
        sr_x_restored = sr_x_restored.loc[sr_x_original_std.index].dropna()

        sr_x_restored = sr_x_restored.loc[
            [idx for idx in sr_x_restored.index if idx not in sr_x_original_start.index]
        ]

        print('sr_x_restored.index')
        print(sr_x_restored.index)
        print('sr_x_restored')
        display(sr_x_restored.head(15))

        
        print('sr_x_original_start to concat')
        display(sr_x_original_start.head(15))

        print('sr_x_restored to concat')
        display(sr_x_restored.head(15))


        sr_x_restored = pd.concat([sr_x_original_start, sr_x_restored.dropna()], axis=0)
        print('sr_x_restored after concat')
        display(sr_x_restored.head(15))
        print('sr_x_restored.index.freq')

        display(list(sr_x_restored.index))

        if sr_x_restored.index[-1] < sr_x_transformed.index[-1]:
            print('!')
            print(sr_x_restored.index[-1])
            print(sr_x_transformed.index[-1])
            sr_x_restored = cls._transform_reverse(sr_x_transformed, sr_x_restored, num_rolling=5)

        return sr_x_restored

    def __init__(self, num_rolling=5):
        self.num_rolling = num_rolling

    def transform_forward(self, sr_x):
        return type(self)._transform_forward(sr_x, self.num_rolling)
    
    def transform_reverse(self, sr_x_transformed, sr_x_original_start):
        return type(self)._transform_reverse(sr_x_transformed, sr_x_original_start, self.num_rolling)