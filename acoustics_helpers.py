import numpy as np
from scipy.optimize import curve_fit

def remove_direct_sound(signal, Fs, signal_length):
    """This method removes the sound that comes from the direct path and also sets the
    maxium signal length to signal_length.
    Parameters
    ----------
    signal : input signal
    Fs : sampling frequency
    signal_length : maximum signal length (trim if needed)
    Returns
    -------
    signal of maxium length of signal_length without the direct path
    """
    part_to_remove = 10/1000 # 10ms
    one_side_window = int(part_to_remove*Fs)
    direct_index = np.argwhere(signal > max(signal)*0.25)[0,0]
    signal_with_direct = signal[max((direct_index - one_side_window), 0):]
    clean_signal = signal_with_direct[2*one_side_window:]
    clean_signal = clean_signal[:int(min(Fs*signal_length, len(clean_signal)))]
    return clean_signal
    
def reweighting(t, sigma, t_cut_off):
    """Function that enables connection of the exponential and constant part of curve
    fiting.
    Parameters
    ----------
    t : x variable of the curve fitting
    sigma : determines the width of the transition region
    t_cut_off : corresponds to the transition point between these behaviours
    Returns
    -------
    weighted hyperbolic tangent function
    """
    return 0.5*(1 - np.tanh((t - t_cut_off)/sigma))
    
def mixing_time(loga, n, e_inf):
    """Function that computes the mixing time out of the learned parameteres.
    The mixing time is derived from the fact that the exponential and constant behaviour have to
    meet at this point.
    Parameters
    ----------
    loga : value of the intersection of the echo density curve at the y axis in log domain
    n : can be ~0 for parallel walls, ~1 for a room without a ceiling and ~2 for a regular room
    e_inf : height at which the echo density curve saturates and reaches constant behaviour
    Returns
    -------
    mixing time parameter
    """
    return np.power(e_inf/np.exp(loga), 1/n)
    
def echo_density_function(logt, a, b, c):
    """Function that is used for the curve fitting.
    Parameters
    ----------
    a = loga : value of the intersection of the echo density curve at the y axis in log domain
    b = n : can be ~0 for parallel walls, ~1 for a room without a ceiling and ~2 for a regular room
    c = e_inf : height at which the echo density curve saturates and reaches constant behaviour    
    signal : 
    Returns
    -------
    set of parameters that provide the lowest fitting loss
    """
    sigma = 0.02
    return np.multiply(reweighting(np.exp(logt), sigma, mixing_time(a, b, c)), (b*logt + a)) + \
           np.multiply(1 - reweighting(np.exp(logt), sigma, mixing_time(a, b, c)), np.log(c))

def curve_fitting_echo_density(data_points_in, Fs):    
    """Fit the estimator for the type of the acoustical space, specifically alpha*t^n on a log(t)
    domain. Our data model consists out of alpha*t^n and a constant segment. The transition between
    these segments is modeled with a sigmoid (hyperbolic tangent) function that acts as a mediator.
    Parameters
    ----------
    data_points_in : sorted density curve for which we need a curve fitting
    Fs : sampling frequency
    Returns
    -------
    parameter that describes the type of the acoustical space
    """
    # Data preprocessing (removing E0 - we try to move the fitting downwards along y axis, so we start at 0)
    index = np.argmin(data_points_in[0:int(Fs/5)])
    data_points_chopped = data_points_in[index:]
    E0 = data_points_chopped[0]    
    data_points = np.maximum(data_points_chopped - E0, 0) + 0.0001
    
    # temporal variable where the curve is fitted
    t = (index + np.array(range(len(data_points))))/Fs
    t_bias = 0.02
    xdata = np.log(t_bias + t)
    ydata = np.log(data_points)
    p0 = np.array([1, 2, 0.9])
    [loga, n, e_inf], pcov = curve_fit(echo_density_function, xdata, ydata, bounds=([-1e10, 0, 0], [1e10, 5, 2]), p0 = p0)
    print(loga, n, e_inf)
    t_mix = mixing_time(loga, n, e_inf)
    a = np.exp(loga)
    return n, np.exp(echo_density_function(np.log(t_bias + t), loga, n, e_inf))+ E0