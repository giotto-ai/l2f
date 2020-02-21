import numpy as np
from scipy.signal import tukey

def generalized_detrending(signal, window_size = 1):
    """This method implements the generalized detrending method that relies on the crest factor.
    It follows the defition from the paper and has an anti-causal nature, since the observation
    window is aligned with the central sample.
    Parameters
    ----------
    signal : input signal
    window_size : number of samples that contribute to the computation
    Returns
    -------
    detrended signal
    """
    detrended_signal = []
    N = 2
    r = 0.5
    w_tukey = tukey(window_size, alpha=r) # r = 0 == rectwin and r = 1 == hann
    signal = np.reshape(signal, (len(signal), 1))
    half_window = round(window_size/2)
    for i in range(len(signal)):        
        il = max(0, (i - half_window))
        ir = min(len(signal), (i + (window_size - half_window)))     
        
        large_signal_segment = signal[il:ir]
        if (len(large_signal_segment) != len(w_tukey)):
            if(ir < len(signal)/2):
                w = w_tukey[max((len(w_tukey) - len(large_signal_segment)), 0):len(w_tukey)]
            else:
                w = w_tukey[0:len(large_signal_segment)]
        else:
            w = w_tukey
        w = np.array([1]) if all(w == 0) else w/sum(w)
        
        large_signal_segment = large_signal_segment**N
        large_segment_mean = np.sum(np.multiply(np.reshape(large_signal_segment, w.shape), w))
        small_signal_segment = signal[i]**N
        
        detrended_signal.append(small_signal_segment/large_segment_mean) # (eq. 1)
    return detrended_signal

def generalized_detrending_causal(signal, window_size = 1):
    """This method implements the generalized detrending method that relies on the crest factor.
    It uses the casal definition, having the current sample aligned with the right edge of the window.
    Rectangular window is used, there is no reweighting.
    Parameters
    ----------
    signal : input signal
    window_size : number of samples that contribute to the computation
    Returns
    -------
    detrended signal
    """
    detrended_signal = []
    N = 2
    signal = np.reshape(signal, (len(signal), 1))
    half_window = round(window_size/2)
    for i in range(len(signal)):        
        il = max(0, (i - half_window))
        ir = min(len(signal), (i + (window_size - half_window)))
        
        large_signal_segment = signal[il:ir]**N
        large_segment_mean = np.sum(large_signal_segment)
        small_signal_segment = signal[ir - 1]**N
        
        detrended_signal.append(small_signal_segment/large_segment_mean) # (eq. 1)
    return detrended_signal

#############################################################################################
def sorted_density_feature(signal, window_size = 1):
    """Sliding window definition of sorted density. Signal is first sorted and then weighted mean
    is computed having as weights the position of samples in the ordered array.
    Parameters
    ----------
    signal : input signal
    window_size : number of samples that contribute to the computation
    Returns
    -------
    feature array of sorted densities
    """
    echo_density = []
    signal = np.reshape(signal, (len(signal), 1))
    half_window = round(window_size/2)
    for i in range(len(signal)):
        il = max(0, (i - half_window))
        ir = min(len(signal), (i + (window_size - half_window)))
        current_density = sorted_density(signal[il:ir])
        echo_density.append(current_density)
    
    return echo_density

def sorted_density(signal):
    """Full-length sorted density feature.
    Parameters
    ----------
    signal : input signal
    Returns
    -------
    full-length sorted density
    """
    t = (np.array(range(len(signal))) + 1)
    signal = np.array(signal)
    signal = signal[signal[:,0].argsort()[::-1]]
    t = np.reshape(t, signal.shape)
    SCT = np.sum(np.multiply(t, signal))/np.sum(signal) # (eq. 2)
    SCT = SCT/(len(signal))#*100 if we want to express in %
    return SCT

def gaussian_sorted_density(ws_detrending):
    """Full-length sorted density feature for Gaussian signal. This is used for the
    normalization of sorted density curve.
    Parameters
    ----------
    ws_detrending : window size for detrending
    Returns
    -------
    sorted density for Gausian signal
    """
    signal_length = 0.2*16000
    NUMBER_OF_TRIALS = 100
    gaussian_SCT = 0
    for i in range(NUMBER_OF_TRIALS):
        signal = np.random.randn(int(signal_length), 1)
        signal = signal/np.max(np.abs(signal))
        detrended_signal = generalized_detrending(signal, ws_detrending)
        echo_density = sorted_density(detrended_signal)
        gaussian_SCT = gaussian_SCT + echo_density
    return gaussian_SCT/NUMBER_OF_TRIALS