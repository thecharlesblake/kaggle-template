import my_util as util
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


# Creates two pipelines, one for the full set of wavelengths, and one for a subset.
def select_and_prep_features(wavelengths):
    print("-----------------------------------")
    print("--- Select and Prepare Features ---")
    print("-----------------------------------\n")

    print("Generating pipelines. One for a subset of wavelengths, and two for single wavelengths ...")

    print("Subset = (evenly spaced) 10% of the wavelengths between 500 and 700nm\n\n")
    selected_wavelengths = np.extract(np.logical_and(wavelengths>=500, wavelengths<=700), wavelengths)[::10]

    return {
        # "all wavelengths": make_pipeline(StandardScaler()),
        "selected wavelength classifier": make_pipeline(util.DataFrameSelector(selected_wavelengths), StandardScaler()),
        "653.930 nm classifier": make_pipeline(util.DataFrameSelector([653.930]), StandardScaler()),
        "428.756 nm classifier": make_pipeline(util.DataFrameSelector([428.756]), StandardScaler()),
    }
