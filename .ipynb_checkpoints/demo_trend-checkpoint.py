import pandas as pd
import numpy as np
from GeneralizedDetrending import GeneralizedDetrending


ts = pd.DataFrame([0, 1, 2, 3, 4, 5])
print("Time series: ", ts)

gnrl_dtr = GeneralizedDetrending(window_size=2)
print("Generalized detrending: ", gnrl_dtr.fit_transform(ts))