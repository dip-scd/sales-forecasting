import unittest
from src.features.timeseries_transform import TimeseriesTransformerDiffDividedByStd
from src.utils.timeseries_utils import generate_test_timeseries
import numpy as np


class TestTimeseriesTransformerDiffDividedByStd(unittest.TestCase):

    def setUp(self):
        self.transform = TimeseriesTransformerDiffDividedByStd(5)

        self.sr_test_data = generate_test_timeseries(
            length=150, 
            mean_trend=2., 
            variance_trend = 5,
            period_freq=3.,
            noise_std=0.3,
            start='2000.01.01',
            freq='D',
            random_state=42
        )

    def test_forward_reverse_equality(self):

        sr_test_data_transformed = self.transform.transform_forward(self.sr_test_data)
        num_rolling_period = self.transform.num_rolling

        sr_test_restored = self.transform.transform_reverse(
            self.sr_test_data[:num_rolling_period],
            sr_test_data_transformed[num_rolling_period:])

        # comparing the original and restored data. 
        # allowing small differences due to floating point arithmetic during the transformation.

        # comparing the original and restored data. 
        # allowing small differences due to floating point arithmetic during the transformation.

        print((self.sr_test_data - sr_test_restored))

        num_comparison_result = ((self.sr_test_data - sr_test_restored).apply(lambda x: np.round(x, 8)) == 0).all()
        # print(num_comparison_result)
        self.assertEqual(num_comparison_result, True, 'Original and restored data are not equal.')


if __name__ == '__main__':
    unittest.main()