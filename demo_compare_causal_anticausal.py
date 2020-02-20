import pandas as pd
import numpy as np
import pandas.util.testing as testing
from SortedDensity import SortedDensity
from CrestFactorDetrending import CrestFactorDetrending

from acoustics import generalized_detrending, generalized_detrending_causal, sorted_density, sorted_density_feature, gaussian_sorted_density

df = pd.DataFrame(np.random.rand(10))
signal = df.to_numpy()[:,0]

ws_detrending = 2
detrending_feature = CrestFactorDetrending(window_size=ws_detrending, is_causal = True) # causal
detrended_df = detrending_feature.fit_transform(df)
detrended_signal = generalized_detrending_causal(signal, ws_detrending)                 # MovingCustomFeature #1

ws_sorted_density = 2
density_feature = SortedDensity(window_size=ws_sorted_density, is_causal = True)        # causal
density_df = density_feature.fit_transform(detrended_df)
echo_density = sorted_density_feature(detrended_signal, ws_sorted_density)              # MovingCustomFeature #2

print(density_df.to_numpy()[:,0])
print(echo_density)