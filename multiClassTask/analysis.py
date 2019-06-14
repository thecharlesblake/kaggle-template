import my_util as util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyse(x_train, y_train, run_opts):
    print("-------------------------------")
    print("--- Analysing Training Data ---")
    print("-------------------------------\n")

    colours = {0: 'blue', 1: 'green', 2: 'pink', 3: 'red', 4: 'gold'}
    xy_data = pd.concat([x_train, y_train], axis=1)

    print("Generating plot for colour ratio...")
    ax = y_train['colour'].value_counts(ascending=True).plot(kind='bar', color=colours.values(),
                                                             title='Number of samples of each colour (training set)')
    ax.set_ylabel("number of samples")
    util.plot('colour_ratio', run_opts, False)

    # Generates a line graph for all of the wavelength samples
    print("\nGenerating wavelength line graph...")
    ax = x_train.iloc[:, ::10].transpose().plot(legend=False, color='brown', alpha=0.1, title='All wavelength values')
    ax.set_ylabel("reflectance intensity")
    ax.set_xlabel("wavelength (nm)")
    util.plot('wavelength_line_graph', run_opts, False)

    # For both the clean and noisy versions of the dataset, plots the line and mean scatter graphs
    noise = np.random.normal(0, 100, x_train.shape)
    for data_type, x_data in {'clean': x_train, 'noisy': x_train + noise}.items():
        print("\nGenerating colour-coded wavelength line graph ("+data_type+")...")
        _, ax = plt.subplots()
        x = x_data.columns.values
        for idx, y in x_data.iterrows():
            ax.plot(x, y, color=colours[y_train.loc[idx][0]], alpha=0.1, linewidth=0.5)
        ax.set_title("Colour-coded wavelength values (" + data_type + ' data)')
        ax.set_ylabel("reflectance intensity")
        ax.set_xlabel("wavelength (nm)")
        util.plot('cc_wavelength_line_graph_'+data_type, run_opts, False)

        print("Generating scatter plot for low vs high wavelength means ("+data_type+")...")
        low_wl_means  = x_data.loc[:, 550.143:600.688].mean(axis=1)
        high_wl_means = x_data.loc[:, 600.688:650.763].mean(axis=1)
        _, ax = plt.subplots()
        ax.set_title('Colour distribution for low vs high wavelength means')
        ax.set_xlabel("mean intensity for range 550-600 nm")
        ax.set_ylabel("mean intensity for range 600-650 nm")
        ax.scatter(low_wl_means, high_wl_means, color=y_train['colour'].apply(lambda x: colours[x]), alpha=0.3,
                   marker='o')
        util.plot('high_low_mean_scatter_'+data_type, run_opts, False)
