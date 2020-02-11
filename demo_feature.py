import pandas as pd
import numpy as np
from gtime.feature_extraction import MovingCustomFunction
from gtime.feature_extraction import MovingAverage
from SortedDensity import SortedDensity

def sorted_density(signal, Fs = 1):
    """Full-length sorted density feature.
    Parameters
    ----------
    signal : input signal
    Fs : sampling frequency
    Returns
    -------
    full-length sorted density
    """
    t = (np.array(range(len(signal))) + 1)/Fs
    signal = np.array(signal)
    signal = signal[signal.argsort()[::-1]]
    t = np.reshape(t, signal.shape)
    SCT = np.sum(np.multiply(t, signal))/np.sum(signal) # (eq. 2)
    SCT = SCT/(len(signal)/Fs)#*100 # if we want to express in %
    return SCT

ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
print("Time series: ", ts)
'''
mv_avg = MovingAverage(window_size=2)
print("Moving average: ", mv_avg.fit_transform(ts))

mv_custom = MovingCustomFunction(np.max, window_size=2)
print("Moving custom function - max: ", mv_custom.fit_transform(ts))

mv_custom = MovingCustomFunction(np.min, window_size=2)
print("Moving custom function - min: ", mv_custom.fit_transform(ts))
'''
srt_dns = SortedDensity(window_size=2)
print("Sorted density: ", srt_dns.fit_transform(ts))

mv_custom = MovingCustomFunction(sorted_density, window_size=2)
print("Moving custom function - sorted_density_feature: ", mv_custom.fit_transform(ts))