import my_util as util
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


# Creates two pipelines, one for the full set of wavelengths, and one for a subset.
def select_and_prep_features(wavelengths):
    print("-----------------------------------")
    print("--- Select and Prepare Features ---")
    print("-----------------------------------\n")

    print("Generating pipelines...\n")

    selected_wavelengths = np.extract(np.logical_and(wavelengths>=550.143, wavelengths<=650.059), wavelengths)
    split_mean_info = {'550-600nm mean': (550.143, 600.330), '600-650nm mean': (600.330, 650.059)}

    return {
        #"all wavelengths classifier": make_pipeline(StandardScaler()),
        "selected wavelength classifier": make_pipeline(util.DataFrameSelector(selected_wavelengths), StandardScaler()),
        "low-high means classifier": make_pipeline(util.DataFrameMeanSelector(split_mean_info), StandardScaler()),
        "noisy data classifier": make_pipeline(util.DataFrameNoiseGenerator(0, 100), StandardScaler())
    }
