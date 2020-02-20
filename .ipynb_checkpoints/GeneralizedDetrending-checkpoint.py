import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from giottotimemaster.gtime.base import FeatureMixin, add_class_name
# will become part of the PREPROCESSING giotto-time module:
# https://github.com/giotto-ai/giotto-time/tree/master/gtime/preprocessing

class GeneralizedDetrending(BaseEstimator, TransformerMixin, FeatureMixin):
    """Generalized detrending model.
    This class removes the trend from the data by using the crest factor definition.
    Each sample is normalize by its weighted surrounding.
    Generalized detrending is defined in (eq. 1) of: H. P. Tukuljac, V. Pulkki,
    H. Gamper, K. Godin, I. J. Tashev and N. Raghuvanshi, "A Sparsity Measure for Echo 
    Density Growth in General Environments," ICASSP 2019 - 2019 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United 
    Kingdom, 2019, pp. 1-5.
    Parameters
    ----------
    window_size : int, optional, default: ``1``
        The number of previous points on which to compute the generalized detrending.
    
    window_size :
    Examples
    >>> import pandas as pd
    >>> from GeneralizedDetrending import GeneralizedDetrending    
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5]) 
    >>> gnrl_dtr = GeneralizedDetrending(window_size=2)  
    >>> gnrl_dtr.fit_transform(ts)
    Generalized detrending:     0__GeneralizedDetrending
    0                       NaN
    1                  1.000000
    2                  0.800000
    3                  0.692308
    4                  0.640000
    5                  0.609756
    --------
    """

    def __init__(self, window_size: int = 1):
        super().__init__()
        self.window_size = window_size

    def fit(self, X, y=None):
        """Fit the estimator.
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Input data.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        self.columns_ = X.columns.values
        return self

    @add_class_name
    def transform(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """Compute the general detrending.
        Parameters
        ----------
        time_series : pd.DataFrame, shape (n_samples, 1), required
            The DataFrame on which to compute the rolling moving average
        Returns
        -------
        time_series_t : pd.DataFrame, shape (n_samples, 1)
            A DataFrame, with the same length as ``time_series``, containing the rolling
            sorted density for each observed time window.
        """
        check_is_fitted(self)
        def detrend(signal):
            import numpy as np
            N = 2
            signal = np.array(signal)
            large_signal_segment = signal**N
            large_segment_mean = np.sum(large_signal_segment)
            small_signal_segment = signal[-1]**N
            return small_signal_segment/large_segment_mean # (eq. 1)
        
        time_series_gnrl_dtr = time_series.rolling(self.window_size).apply(detrend, raw=False)
        return time_series_gnrl_dtr