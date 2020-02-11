import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from giottotimemaster.gtime.base import FeatureMixin, add_class_name
# will become part of the feature extraction module of giotto-time:
# https://github.com/giotto-ai/giotto-time/blob/master/gtime/feature_extraction/standard.py

class SortedDensity(BaseEstimator, TransformerMixin, FeatureMixin):
    """For each row in ``time_series``, compute the sorted density function of the
    previous ``window_size`` rows. If there are not enough rows, the value is ``Nan``.
    Sorted density measured is defined in (eq. 1) of: H. P. Tukuljac, V. Pulkki,
    H. Gamper, K. Godin, I. J. Tashev and N. Raghuvanshi, "A Sparsity Measure for Echo 
    Density Growth in General Environments," ICASSP 2019 - 2019 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United 
    Kingdom, 2019, pp. 1-5.
    Parameters
    ----------
    window_size : int, optional, default: ``1``
        The number of previous points on which to compute the sorted density.
    Examples
    --------
    >>> import pandas as pd
    >>> from gtime.feature_extraction import SortedDensity
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
    >>> mv_avg = SortedDensity(window_size=2)
    >>> mv_avg.fit_transform(ts)
       0__SortedDensity
    0                      NaN
    1                 0.500000
    2                 0.666667
    3                 0.700000
    4                 0.714286
    5                 0.722222
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
        """Compute the sorted density, for every row of ``time_series``, of the previous
        ``window_size`` elements.
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
        def sorted_density(signal, Fs = 1):
            import numpy as np
            t = (np.array(range(len(signal))) + 1)/Fs
            signal = np.array(signal)
            signal = signal[signal.argsort()[::-1]]
            t = np.reshape(t, signal.shape)
            SD = np.sum(np.multiply(t, signal))/np.sum(signal) # (eq. 2)
            SD = SD/(len(signal)/Fs)
            return SD
        
        time_series_srt_dns = time_series.rolling(self.window_size).apply(sorted_density, raw=False)
        return time_series_srt_dns