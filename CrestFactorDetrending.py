from gtime.feature_extraction import MovingCustomFunction

class CrestFactorDetrending(MovingCustomFunction):
    """Crest factor detrending model.
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
        The number of previous points on which to compute the crest factor detrending.    
    is_causal : bool, optional, default: ``True``
        Whether the current sample is computed based only on the past or also on the future.
    Examples
    >>> import pandas as pd
    >>> from CrestFactorDetrending import CrestFactorDetrending    
    >>> ts = pd.DataFrame([0, 1, 2, 3, 4, 5]) 
    >>> gnrl_dtr = CrestFactorDetrending(window_size=2)  
    >>> gnrl_dtr.fit_transform(ts)
       0__CrestFactorDetrending
    0                       NaN
    1                  1.000000
    2                  0.800000
    3                  0.692308
    4                  0.640000
    5                  0.609756
    --------
    """

    def __init__(self, window_size: int = 1, is_causal: bool = False):
        super().__init__(self.detrend)
        self.window_size = window_size
        self.is_causal = is_causal
    
    def detrend(self, signal):
        import numpy as np
        N = 2
        signal = np.array(signal)
        large_signal_segment = signal**N
        large_segment_mean = np.sum(large_signal_segment)
        if (self.is_causal):
            ref_index = -1
        else:
            ref_index = int(len(signal)/2) 
        small_signal_segment = signal[ref_index]**N
        return small_signal_segment/large_segment_mean # (eq. 1)