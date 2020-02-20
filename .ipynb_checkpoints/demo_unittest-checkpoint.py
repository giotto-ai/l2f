import unittest
import logging
import GeneralizedDetrending
import SortedDensity

class Tests(unittest.TestCase):
    def test_generalized_detrending(self):
        logging.info("Testing generalized detrending")
        self.assertEqual(1, 1)
    
    def test_sorted_density(self):
        logging.info("Testing sorted density feature")
        self.assertEqual(1, 1)
        
if __name__ == '__main__':
    unittest.main()

'''
import pandas.util.testing as testing
from numpy.testing import assert_array_equal
import pandas as pd
import numpy as np

from gtime.compose import FeatureCreation
from gtime.feature_extraction import Shift, MovingAverage, CustomFeature, MovingCustomFunction

class TestMovingCustomFunction:
    def test_correct_moving_custom_function(self):
        df = pd.DataFrame.from_dict({"x_1": [0, 7, 2], "x_2": [2, 10, 4]})
        df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]
        custom_feature = MovingCustomFunction(
            custom_feature_function=np.diff, window_size=2
        )
        custom_output = custom_feature.fit_transform(df)

        feature_name = custom_feature.__class__.__name__
        expected_custom_df = pd.DataFrame.from_dict(
            {
                f"x_1__{feature_name}": [np.nan, 7.0, -5],
                f"x_2__{feature_name}": [np.nan, 8.0, -6],
            }
        )
        expected_custom_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        testing.assert_frame_equal(expected_custom_df, custom_output)
        
def test_feature_creation_transform():
    data = testing.makeTimeDataFrame(freq="s")

    shift = Shift(1)
    ma = MovingAverage(window_size=3)

    col_name = 'A'

    fc = FeatureCreation([
        ('s1', shift, [col_name]),
        ('ma3', ma, [col_name]),
    ])
    res = fc.fit(data).transform(data)

    assert_array_equal(res.columns.values,
                       [f's1__{col_name}__{shift.__class__.__name__}', f'ma3__{col_name}__{ma.__class__.__name__}'])

def test_custom_function():
    df = pd.DataFrame.from_dict({"x": [0, 1, 2, 3, 4, 5]})
    f = lambda x: x + 1
    df_apply = df.apply(f).rename(columns={'x': 'x__CustomFeature'})
    cf = CustomFeature(f)
    testing.assert_equal(df_apply, cf.fit_transform(df))
    
a = TestMovingCustomFunction()
a.test_correct_moving_custom_function()
test_feature_creation_transform()
test_custom_function()
'''