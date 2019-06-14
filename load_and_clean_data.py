import pandas as pd
import os


def load_and_clean_data(data_path):
    print("---------------------------------------")
    print("--- Loading and Cleaning Input Data ---")
    print("---------------------------------------\n")

    # Reads in data from CSV
    print("Reading in data from CSV...\n")

    print("Adding headers to loaded data...\n")
    wavelength_data = pd.read_csv(os.path.join(data_path, "Wavelength.csv"), header=None).values.flatten()
    x_data = pd.read_csv(os.path.join(data_path, "X.csv"), names=wavelength_data)
    x_to_classify_data = pd.read_csv(os.path.join(data_path, "XToClassify.csv"), names=wavelength_data)
    y_data = pd.read_csv(os.path.join(data_path, "y.csv"), names=['colour'])

    # Prints basic summary
    print("--- Basic summary of wavelength data ---")
    print(x_data.info(), end='\n\n')
    print("--- Basic summary of colour data ---")
    print(y_data.info(), end='\n\n')

    # Prints some descriptive statistics for the dataset
    print("--- Descriptive statistics for wavelength data ---")
    print(x_data.describe(include='all'), end='\n\n')
    print("--- Descriptive statistics for colour data ---")
    print(y_data.describe(include='all'), end='\n\n')

    print("--- 10 columns with largest max value in wavelength data ---")
    maxes = x_data.max()
    print(maxes.nlargest(10), end='\n\n')
    print("--- 10 columns with smallest min value in wavelength data ---")
    mins = x_data.min()
    print(mins.nsmallest(10), end='\n\n')
    stds =x_data.std()
    print("--- 10 columns with largest std in wavelength data ---")
    print(stds.nlargest(10), end='\n\n')
    print("--- 10 columns with smallest std in wavelength data ---")
    print(stds.nsmallest(10), end='\n\n')

    return x_data, y_data, x_to_classify_data
