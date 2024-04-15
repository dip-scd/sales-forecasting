import unittest
from src.features.make_model_dataset import create_lag_features
from src.utils.timeseries_utils import generate_test_timeseries
import numpy as np
import pandas as pd

class TestFeaturesCreation(unittest.TestCase):

    def setUp(self):
        sr_test_data = generate_test_timeseries(
            length=10, 
            mean_trend=2., 
            variance_trend = 1,
            period_freq=0.,
            noise_std=0.0,
            start='2000.01.01',
            freq='D',
            random_state=42
        )
        self.df_test_data = pd.DataFrame({
            'x': sr_test_data,
        }, index=sr_test_data.index)

    def test_create_lag_features(self):

        max_lag = 9
        df_test_data_with_lags = create_lag_features(
            self.df_test_data, max_lag=max_lag)

        arr_x_0_col = df_test_data_with_lags[('x', '-0')].values[-max_lag:]
        arr_x_last_row = df_test_data_with_lags['x'].iloc[-1].values[-max_lag:]
        comaprison_result = (arr_x_0_col == arr_x_last_row).all()    
        self.assertEqual(comaprison_result, True, 'Created lagged values are not correct')


if __name__ == '__main__':
    unittest.main()