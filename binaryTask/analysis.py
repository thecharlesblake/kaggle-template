import my_util as util
import pandas as pd
import matplotlib.pyplot as plt


def analyse(x_train, y_train, run_opts):
    print("-------------------------------")
    print("--- Analysing Training Data ---")
    print("-------------------------------\n")

    colours = {0: 'green', 1: 'red'}
    xy_data = pd.concat([x_train, y_train], axis=1)

    print("Generating wavelength KDE...\n")
    ax = x_train.iloc[:, ::10].plot(kind='kde', colormap='cool', legend=False, title='Wavelength values KDE')
    ax.set_xlabel("reflectance intensity")
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 0.3)
    util.plot('wavelength_kde', run_opts, True)

    print("Generating wavelength line graph...\n")
    ax = x_train.iloc[:, ::10].transpose().plot(legend=False, color='brown', alpha=0.1, title='All wavelength values')
    ax.set_ylabel("reflectance intensity")
    ax.set_xlabel("wavelength (nm)")
    util.plot('wavelength_line_graph', run_opts, True)

    print("Generating colour-coded wavelength line graph...\n")
    _, ax = plt.subplots()
    x = xy_data.drop('colour', axis=1).columns.values
    for _, row in xy_data.iterrows():
        y = row.drop('colour')
        ax.plot(x, y, color=colours[row['colour']], alpha=0.1, linewidth=0.5)
    ax.set_title("Colour-coded wavelength values")
    ax.set_ylabel("reflectance intensity")
    ax.set_xlabel("wavelength (nm)")
    util.plot('cc_wavelength_line_graph', run_opts, True)

    print("Generating plot for green/red ratio...\n")
    ax = y_train['colour'].value_counts(ascending=True).plot(kind='bar', color=['green', 'red'],
                                                             title='Number of samples of each colour (training set)')
    ax.set_ylabel("number of samples")
    util.plot('green_red_ratio', run_opts, True)

    # Calculates the correlation of each pair of input features
    print("--- Wavelength correlation ---")
    corr_matrix = xy_data.corr().abs()

    # Generates a plot of the correlation of each wavelength feature to the colour
    appliances_corr = corr_matrix['colour'].sort_values(ascending=True)
    appliances_corr.drop('colour', inplace=True)
    print("--- 15 wavelengths with largest correlation to colour ---\n")
    print(appliances_corr.nlargest(15), end='\n\n')
    print("--- 15 wavelengths with smallest correlation to colour ---\n")
    print(appliances_corr.nsmallest(15), end='\n\n')

    print("Generating scatter plot for most correlated wavelength with colour...\n")
    _, ax = plt.subplots()
    ax.scatter(xy_data[653.930], xy_data['colour'], color=xy_data['colour'].apply(lambda x: colours[x]), alpha=0.2,
               marker='s')
    ax.set_title('Colour correlation for most correlated wavelength')
    ax.set_xlabel("reflectance intensity")
    util.plot('most_correlated_scatter', run_opts, True)

    _, ax = plt.subplots()
    ax.scatter(xy_data[428.756], xy_data['colour'], color=xy_data['colour'].apply(lambda x: colours[x]), alpha=0.2,
               marker='s')
    ax.set_title('Colour correlation for least correlated wavelength')
    ax.set_xlabel("reflectance intensity")
    util.plot('least_correlated_scatter', run_opts, True)
