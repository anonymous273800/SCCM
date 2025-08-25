import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from Utils import Constants


def plot_results(x_axis, y_axis1, y_axis2, kpi, label1, label2, drift_location, log_enabled,
                 legend_loc, drift_type, gradual_drift_locations, gradual_drift_concepts):
    # Plotting the data
    length = np.min([len(arr) for arr in [x_axis, y_axis1, y_axis2]])
    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]

    if log_enabled:
        # Log transform the data
        y_axis1 = np.log(y_axis1 + 1)  # Adding 1 to avoid log(0)
        y_axis2 = np.log(y_axis2 + 1)  # Adding 1 to avoid log(0)

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    line1, = plt.plot(x_axis, y_axis1, linestyle='-', marker='.', markersize=1, linewidth=0.5, label=label1)
    line2, = plt.plot(x_axis, y_axis2, linestyle='-', marker='.', markersize=1, linewidth=0.5,
                      label=label2)

    if drift_type == 'abrupt':
        # Shade the region from x=500 onwards
        plt.axvspan(drift_location, max(x_axis), color=Constants.color_yello, alpha=0.3, label='Concept Drift')

    if drift_type == 'incremental':
        # Draw a vertical yellow line at each drift_location
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=.6, label=None)

        concept_drift_line = Line2D([0], [0], color=Constants.color_yello, linestyle='-', linewidth=.6,
                                    label='Concept Drift')

    if drift_type == 'gradual':
        # gradual_drift_location = [250, 350, 450, 650, 750, 1000]
        # concepts = ['c1', 'c2', 'c1', 'c2', 'c1', 'c2']
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)  # Default color if concept not found
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        # Create custom legend entries
        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()]

    # Adding labels and title
    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2': plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE': plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    # Adjust font size of numbers on x-axis and y-axis
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    # Customize grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Only vertical lines with dashed style

    # Remove top and right spines
    sns.despine()

    # Adding grid
    plt.grid(True)

    # Adding legend
    if drift_type == 'incremental':
        plt.legend(handles=[line1, line2, concept_drift_line], fontsize='small', loc=legend_loc, fancybox=True,
                   shadow=True,
                   borderpad=1, labelspacing=.5,
                   facecolor=Constants.color_light_blue, edgecolor=Constants.color_black)

        # Adding legend
    if drift_type == 'gradual':
        plt.legend(handles=[line1, line2, gradual_concept_drift_line[0], gradual_concept_drift_line[1]],
                   fontsize='small',
                   loc=legend_loc, fancybox=True, shadow=True,
                   borderpad=1, labelspacing=.5,
                   facecolor='lightblue', edgecolor=Constants.color_black)

    if drift_type == 'abrupt':
        plt.legend(fontsize='small', loc=legend_loc, fancybox=True, shadow=True, borderpad=1, labelspacing=.5,
                   facecolor='lightblue', edgecolor=Constants.color_black)

    # Show plot
    plt.show()
