import numpy as np
import pandas as pd
import h5py

from SortedDensity import SortedDensity
from CrestFactorDetrending import CrestFactorDetrending

from acoustics import generalized_detrending, sorted_density, sorted_density_feature, gaussian_sorted_density
from acoustics_helpers import remove_direct_sound, curve_fitting_echo_density

# 3 demos available
with h5py.File('data.h5', 'r') as hf:
    Fs_array = list(hf['Fs'])
    dataset_simulation_polybox = list(hf['dataset_simulation_polybox'])
    dataset_simulation_sliding_lid = list(hf['dataset_simulation_sliding_lid'])
    dataset_measurement_volume = list(hf['dataset_measurement_volume'])
    
dictionary = {'closed' : dataset_simulation_sliding_lid[0], 
              'almost_closed' : dataset_simulation_sliding_lid[1], 
              'almost_open' : dataset_simulation_sliding_lid[2], 
              'open' : dataset_simulation_sliding_lid[3]}
Fs = Fs_array[1]
data_simulation_sliding_lid = pd.DataFrame(data = dictionary, index = pd.timedelta_range(start=pd.Timedelta(days=0), freq=str(round(1000/Fs, 5)) + 'ms', 
                                   periods=len(dataset_simulation_sliding_lid[0])))
data_simulation_sliding_lid.head()

############################## DATA LOADED ##############################

# ws stands for window size
ws_detrending = int(0.025*Fs)     # 25ms
ws_sorted_density = int(0.2*Fs)   # 200ms
milliseconds_to_remove = 10 # for the remove the direct path function
signal_length = 1

gaussian_sd = gaussian_sorted_density(ws_detrending, Fs)
print('Sorted density for Gaussian signal: ', gaussian_sd)

signal = data_simulation_sliding_lid['closed'].to_numpy()
t = np.array(range(len(signal)))/Fs
signal = remove_direct_sound(signal, Fs, signal_length)

detrended_signal = generalized_detrending(signal, ws_detrending, Fs)                            # MovingCustomFeature #1
echo_density = sorted_density_feature(detrended_signal, ws_sorted_density, Fs)                  # MovingCustomFeature #2

echo_density = echo_density/gaussian_sd
n = curve_fitting_echo_density(echo_density, Fs)
print(n)