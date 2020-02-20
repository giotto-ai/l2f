import pandas as pd
import numpy as np
import pandas.util.testing as testing
from SortedDensity import SortedDensity
from CrestFactorDetrending import CrestFactorDetrending
    
class TestCrestFactorDetrending:
    df = pd.DataFrame.from_dict({"x_1": [0, 7, 2], "x_2": [2, 10, 4]})
    df.index = [
        pd.Timestamp(2000, 1, 1),
        pd.Timestamp(2000, 2, 1),
        pd.Timestamp(2000, 3, 1),
    ]
    
    def test_causal(self):
        custom_feature = CrestFactorDetrending(window_size=2, is_causal = True) # causal
        custom_output = custom_feature.fit_transform(self.df)

        feature_name = custom_feature.__class__.__name__
        expected_custom_df = pd.DataFrame.from_dict(
            {
                f"x_1__{feature_name}": [np.nan, 1.0, 0.07547169811320754],
                f"x_2__{feature_name}": [np.nan, 0.9615384615384616, 0.13793103448275862],
            }
        )
        expected_custom_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        testing.assert_frame_equal(expected_custom_df, custom_output)        
        print("\n\nCAUSAL TEST")
        print(self.df)
        print(expected_custom_df)
        print(custom_output)
        
    def test_anticausal(self):
        custom_feature = CrestFactorDetrending(window_size=2, is_causal = False) # anticausal
        custom_output = custom_feature.fit_transform(self.df)

        feature_name = custom_feature.__class__.__name__
        expected_custom_df = pd.DataFrame.from_dict(
            {
                f"x_1__{feature_name}": [np.nan, 1.0, 0.07547169811320754],
                f"x_2__{feature_name}": [np.nan, 0.9615384615384616, 0.13793103448275862],
            }
        )
        expected_custom_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        testing.assert_frame_equal(expected_custom_df, custom_output)        
        print("\n\nANTICAUSAL TEST")
        print(self.df)
        print(expected_custom_df)
        print(custom_output)
        
class TestSortedDensity:
    df = pd.DataFrame.from_dict({"x_1": [0, 7, 2], "x_2": [2, 10, 4]})
    df.index = [
        pd.Timestamp(2000, 1, 1),
        pd.Timestamp(2000, 2, 1),
        pd.Timestamp(2000, 3, 1),
    ]
    
    def test_causal(self):
        custom_feature = SortedDensity(window_size=2, is_causal = True) # causal
        custom_output = custom_feature.fit_transform(self.df)

        feature_name = custom_feature.__class__.__name__
        expected_custom_df = pd.DataFrame.from_dict(
            {
                f"x_1__{feature_name}": [np.nan, 0.5, 0.6111111111111112],
                f"x_2__{feature_name}": [np.nan, 0.5833333333333334, 0.6428571428571429],
            }
        )
        expected_custom_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        testing.assert_frame_equal(expected_custom_df, custom_output)
        print("\n\nCAUSAL TEST")
        print(self.df)
        print(expected_custom_df)
        print(custom_output)
        
    def test_anticausal(self):
        custom_feature = SortedDensity(window_size=2, is_causal = False) # anticausal
        custom_output = custom_feature.fit_transform(self.df)

        feature_name = custom_feature.__class__.__name__
        expected_custom_df = pd.DataFrame.from_dict(
            {
                f"x_1__{feature_name}": [np.nan, 0.5, 0.6111111111111112],
                f"x_2__{feature_name}": [np.nan, 0.5833333333333334, 0.6428571428571429],
            }
        )
        expected_custom_df.index = [
            pd.Timestamp(2000, 1, 1),
            pd.Timestamp(2000, 2, 1),
            pd.Timestamp(2000, 3, 1),
        ]

        testing.assert_frame_equal(expected_custom_df, custom_output)
        print("\n\nANTICAUSAL TEST")
        print(self.df)
        print(expected_custom_df)
        print(custom_output)


cf_dtr = TestCrestFactorDetrending()
cf_dtr.test_causal()
cf_dtr.test_anticausal()

srt_dns = TestSortedDensity()
srt_dns.test_causal()
srt_dns.test_anticausal()
