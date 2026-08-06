from turtle import color

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import os
from Utils import Constants


def plot_results(
    x_axis,
    olr_wa_acc_list,
    olr_wa_mse_list,
    olr_wa_sccm_acc_list,
    olr_wa_sccm_mse_list,
    kpi,
    label1,
    label2,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts
):
    if kpi == 'R2':
        y_axis1 = olr_wa_acc_list
        y_axis2 = olr_wa_sccm_acc_list
    else:  # kpi == 'MSE':
        y_axis1 = olr_wa_mse_list
        y_axis2 = olr_wa_sccm_mse_list

    length = np.min([len(arr) for arr in [x_axis, y_axis1, y_axis2]])
    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(
        x_axis, y_axis1,
        linestyle='-',
        marker='o',
        markersize=2,
        linewidth=1.0,
        label=label1,
        zorder=3
    )

    line2, = plt.plot(
        x_axis, y_axis2,
        linestyle='--',
        marker='s',
        markersize=2,
        linewidth=1.0,
        label=label2,
        zorder=4
    )

    if drift_type == 'abrupt':
        plt.axvspan(
            drift_location,
            max(x_axis),
            color=Constants.color_yello,
            alpha=0.3,
            label='Concept Drift'
        )

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=0.6, label=None)

        concept_drift_line = Line2D(
            [0], [0],
            color=Constants.color_yello,
            linestyle='-',
            linewidth=0.6,
            label='Concept Drift'
        )

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(
            handles=[line1, line2, concept_drift_line],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.5,
            facecolor=Constants.color_light_blue,
            edgecolor=Constants.color_black
        )

    elif drift_type == 'gradual':
        plt.legend(
            handles=[line1, line2, gradual_concept_drift_line[0], gradual_concept_drift_line[1]],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    else:  # abrupt
        plt.legend(
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    plt.show()


def plot_results_simple(x_axis, y_axis1, y_axis2, kpi, label1, label2, log_enabled, legend_loc):
    length = np.min([len(arr) for arr in [x_axis, y_axis1, y_axis2]])
    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(
        x_axis, y_axis1,
        linestyle='-',
        marker='o',
        markersize=2,
        linewidth=1.1,
        label=label1,
        zorder=3
    )

    line2, = plt.plot(
        x_axis, y_axis2,
        linestyle='--',
        marker='s',
        markersize=2,
        linewidth=1.1,
        label=label2,
        zorder=4
    )

    plt.xlabel('$N$', fontsize=8)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=8)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=8)

    plt.title('Performance Comparison', fontsize=8)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    plt.legend(
        fontsize='small',
        loc=legend_loc,
        fancybox=True,
        shadow=True,
        borderpad=1,
        labelspacing=0.5,
        facecolor='lightblue',
        edgecolor=Constants.color_black
    )

    plt.show()


def plot_results_three_models(
    x_axis,
    y_axis1,
    y_axis2,
    y_axis3,
    kpi,
    label1,
    label2,
    label3,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts
):
    length = np.min([len(arr) for arr in [x_axis, y_axis1, y_axis2, y_axis3]])
    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]
    y_axis3 = y_axis3[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(
        x_axis, y_axis1,
        linestyle='-',
        marker='o',
        markersize=2,
        linewidth=1.0,
        label=label1,
        zorder=2
    )

    line2, = plt.plot(
        x_axis, y_axis2,
        linestyle='--',
        marker='s',
        markersize=2,
        linewidth=1.0,
        label=label2,
        zorder=3
    )

    line3, = plt.plot(
        x_axis, y_axis3,
        linestyle=':',
        marker='^',
        markersize=3,
        markevery=3,
        linewidth=1.6,
        label=label3,
        zorder=4
    )

    if drift_type == 'abrupt':
        plt.axvspan(
            drift_location,
            max(x_axis),
            color=Constants.color_yello,
            alpha=0.3,
            label='Concept Drift'
        )

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=0.6, label=None)

        concept_drift_line = Line2D(
            [0], [0],
            color=Constants.color_yello,
            linestyle='-',
            linewidth=0.6,
            label='Concept Drift'
        )

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(
            handles=[line1, line2, line3, concept_drift_line],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.5,
            facecolor=Constants.color_light_blue,
            edgecolor=Constants.color_black
        )

    elif drift_type == 'gradual':
        plt.legend(
            handles=[line1, line2, line3, gradual_concept_drift_line[0], gradual_concept_drift_line[1]],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    else:  # abrupt
        plt.legend(
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=0.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    plt.show()


def plot_results_six_models(
    x_axis,
    olr_wa_acc_list,
    olr_wa_mse_list,
    olr_wa_sccm_acc_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_acc_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_acc_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_kswin_reset_r2_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_r2_avg,
    olr_wa_kswin_window_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts,
    save_path
):
    if kpi == 'R2':
        y_axis1 = olr_wa_acc_list
        y_axis2 = olr_wa_sccm_acc_list
        y_axis3 = olr_wa_adwin_reset_acc_list
        y_axis4 = olr_wa_adwin_window_acc_list
        y_axis5 = olr_wa_kswin_reset_r2_avg
        y_axis6 = olr_wa_kswin_window_r2_avg
    else:
        y_axis1 = olr_wa_mse_list
        y_axis2 = olr_wa_sccm_mse_list
        y_axis3 = olr_wa_adwin_reset_mse_list
        y_axis4 = olr_wa_adwin_window_mse_list
        y_axis5 = olr_wa_kswin_reset_mse_avg
        y_axis6 = olr_wa_kswin_window_mse_avg

    length = np.min([len(arr) for arr in [x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5, y_axis6]])
    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]
    y_axis3 = y_axis3[:length]
    y_axis4 = y_axis4[:length]
    y_axis5 = y_axis5[:length]
    y_axis6 = y_axis6[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)

    plt.figure(figsize=(10, 6))
    line1, = plt.plot(x_axis, y_axis1, linestyle='-', marker='o', markersize=1.5, linewidth=0.7, label=label1, color='#4c72b0')
    line2, = plt.plot(x_axis, y_axis2, linestyle='-', marker='^', markersize=1.5, linewidth=1.0, label=label2, color='#dd8452')
    line3, = plt.plot(x_axis, y_axis3, linestyle='--', marker='s', markersize=1.2, linewidth=0.7, label=label3, color='#55a868')
    line4, = plt.plot(x_axis, y_axis4, linestyle=':', marker='D', markersize=1.2, linewidth=0.7, label=label4, color='#8172b3')
    line5, = plt.plot(x_axis, y_axis5, linestyle='-.', marker='v', markersize=1.3, linewidth=0.8, label=label5, color='#2a9d8f')
    line6, = plt.plot(x_axis, y_axis6, linestyle=(0, (5, 1)), marker='X', markersize=1.3, linewidth=0.8, label=label6, color='#8c6d31')

    if drift_type == 'abrupt':
        plt.axvspan(drift_location, max(x_axis), color=Constants.color_yello, alpha=0.3, label='Concept Drift')

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=.6, label=None)

        concept_drift_line = Line2D([0], [0], color=Constants.color_yello, linestyle='-', linewidth=.6,
                                    label='Concept Drift')

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(handles=[line1, line2, concept_drift_line], fontsize='small', loc=legend_loc, fancybox=True,
                   shadow=True, borderpad=1, labelspacing=.5,
                   facecolor=Constants.color_light_blue, edgecolor=Constants.color_black)

    if drift_type == 'gradual':
        plt.legend(handles=[line1, line2, gradual_concept_drift_line[0], gradual_concept_drift_line[1]],
                   fontsize='small', loc=legend_loc, fancybox=True, shadow=True,
                   borderpad=1, labelspacing=.5,
                   facecolor='lightblue', edgecolor=Constants.color_black)

    if drift_type == 'abrupt':
        plt.legend(fontsize='small', loc=legend_loc, fancybox=True, shadow=True, borderpad=1, labelspacing=.5,
                   facecolor='lightblue', edgecolor=Constants.color_black)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()


def plot_results_seven_models(
    x_axis,
    olr_wa_acc_list,
    olr_wa_mse_list,
    olr_wa_sccm_acc_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_acc_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_acc_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_adwin_sspt_r2_avg,
    olr_wa_adwin_sspt_mse_avg,
    olr_wa_kswin_reset_r2_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_r2_avg,
    olr_wa_kswin_window_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts,
    save_path
):
    if kpi == 'R2':
        y_axis1 = olr_wa_acc_list
        y_axis2 = olr_wa_sccm_acc_list
        y_axis3 = olr_wa_adwin_reset_acc_list
        y_axis4 = olr_wa_adwin_window_acc_list
        y_axis5 = olr_wa_adwin_sspt_r2_avg
        y_axis6 = olr_wa_kswin_reset_r2_avg
        y_axis7 = olr_wa_kswin_window_r2_avg
    else:
        y_axis1 = olr_wa_mse_list
        y_axis2 = olr_wa_sccm_mse_list
        y_axis3 = olr_wa_adwin_reset_mse_list
        y_axis4 = olr_wa_adwin_window_mse_list
        y_axis5 = olr_wa_adwin_sspt_mse_avg
        y_axis6 = olr_wa_kswin_reset_mse_avg
        y_axis7 = olr_wa_kswin_window_mse_avg

    length = np.min([
        len(arr) for arr in [
            x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5, y_axis6, y_axis7
        ]
    ])
    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]
    y_axis3 = y_axis3[:length]
    y_axis4 = y_axis4[:length]
    y_axis5 = y_axis5[:length]
    y_axis6 = y_axis6[:length]
    y_axis7 = y_axis7[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)
        y_axis7 = np.log(y_axis7 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(x_axis, y_axis1, linestyle='-', marker='o', markersize=1.5, linewidth=0.7, label=label1, color='#4c72b0')
    line2, = plt.plot(x_axis, y_axis2, linestyle='-', marker='^', markersize=1.5, linewidth=1.0, label=label2, color='#dd8452')
    line3, = plt.plot(x_axis, y_axis3, linestyle='--', marker='s', markersize=1.2, linewidth=0.7, label=label3, color='#55a868')
    line4, = plt.plot(x_axis, y_axis4, linestyle=':', marker='D', markersize=1.2, linewidth=0.7, label=label4, color='#8172b3')
    line5, = plt.plot(x_axis, y_axis5, linestyle=(0, (3, 1, 1, 1)), marker='P', markersize=1.3, linewidth=0.8, label=label5, color='#c44e52')
    line6, = plt.plot(x_axis, y_axis6, linestyle='-.', marker='v', markersize=1.3, linewidth=0.8, label=label6, color='#2a9d8f')
    line7, = plt.plot(x_axis, y_axis7, linestyle=(0, (5, 1)), marker='X', markersize=1.3, linewidth=0.8, label=label7, color='#8c6d31')

    if drift_type == 'abrupt':
        plt.axvspan(drift_location, max(x_axis), color=Constants.color_yello, alpha=0.3, label='Concept Drift')

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=.6, label=None)

        concept_drift_line = Line2D([0], [0], color=Constants.color_yello, linestyle='-', linewidth=.6,
                                    label='Concept Drift')

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(
            handles=[line1, line2, line3, line4, line5, line6, line7, concept_drift_line],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.5,
            facecolor=Constants.color_light_blue,
            edgecolor=Constants.color_black
        )

    if drift_type == 'gradual':
        plt.legend(
            handles=[line1, line2, line3, line4, line5, line6, line7,
                     gradual_concept_drift_line[0], gradual_concept_drift_line[1]],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if drift_type == 'abrupt':
        plt.legend(
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()


def plot_results_eight_models(
    x_axis,
    olr_wa_acc_list,
    olr_wa_mse_list,
    olr_wa_sccm_acc_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_acc_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_acc_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_adwin_sspt_r2_avg,
    olr_wa_adwin_sspt_mse_avg,
    olr_wa_adwin_ohl_r2_avg,
    olr_wa_adwin_ohl_mse_avg,
    olr_wa_kswin_reset_r2_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_r2_avg,
    olr_wa_kswin_window_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    label8,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts,
    save_path
):
    if kpi == 'R2':
        y_axis1 = olr_wa_acc_list
        y_axis2 = olr_wa_sccm_acc_list
        y_axis3 = olr_wa_adwin_reset_acc_list
        y_axis4 = olr_wa_adwin_window_acc_list
        y_axis5 = olr_wa_adwin_sspt_r2_avg
        y_axis6 = olr_wa_adwin_ohl_r2_avg
        y_axis7 = olr_wa_kswin_reset_r2_avg
        y_axis8 = olr_wa_kswin_window_r2_avg
    else:
        y_axis1 = olr_wa_mse_list
        y_axis2 = olr_wa_sccm_mse_list
        y_axis3 = olr_wa_adwin_reset_mse_list
        y_axis4 = olr_wa_adwin_window_mse_list
        y_axis5 = olr_wa_adwin_sspt_mse_avg
        y_axis6 = olr_wa_adwin_ohl_mse_avg
        y_axis7 = olr_wa_kswin_reset_mse_avg
        y_axis8 = olr_wa_kswin_window_mse_avg

    length = np.min([
        len(arr) for arr in [
            x_axis, y_axis1, y_axis2, y_axis3, y_axis4,
            y_axis5, y_axis6, y_axis7, y_axis8
        ]
    ])

    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]
    y_axis3 = y_axis3[:length]
    y_axis4 = y_axis4[:length]
    y_axis5 = y_axis5[:length]
    y_axis6 = y_axis6[:length]
    y_axis7 = y_axis7[:length]
    y_axis8 = y_axis8[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)
        y_axis7 = np.log(y_axis7 + 1)
        y_axis8 = np.log(y_axis8 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(
        x_axis, y_axis1,
        linestyle='-',
        marker='o',
        markersize=1.5,
        linewidth=0.7,
        label=label1,
        color='#4c72b0'
    )

    line2, = plt.plot(
        x_axis, y_axis2,
        linestyle='-',
        marker='^',
        markersize=1.5,
        linewidth=1.0,
        label=label2,
        color='#dd8452'
    )

    line3, = plt.plot(
        x_axis, y_axis3,
        linestyle='--',
        marker='s',
        markersize=1.2,
        linewidth=0.7,
        label=label3,
        color='#55a868'
    )

    line4, = plt.plot(
        x_axis, y_axis4,
        linestyle=':',
        marker='D',
        markersize=1.2,
        linewidth=0.7,
        label=label4,
        color='#8172b3'
    )

    line5, = plt.plot(
        x_axis, y_axis5,
        linestyle=(0, (3, 1, 1, 1)),
        marker='P',
        markersize=1.3,
        linewidth=0.8,
        label=label5,
        color='#c44e52'
    )

    line6, = plt.plot(
        x_axis, y_axis6,
        linestyle=(0, (1, 1)),
        marker='*',
        markersize=2.0,
        linewidth=0.8,
        label=label6,
        color='#64b5cd'
    )

    line7, = plt.plot(
        x_axis, y_axis7,
        linestyle='-.',
        marker='v',
        markersize=1.3,
        linewidth=0.8,
        label=label7,
        color='#2a9d8f'
    )

    line8, = plt.plot(
        x_axis, y_axis8,
        linestyle=(0, (5, 1)),
        marker='X',
        markersize=1.3,
        linewidth=0.8,
        label=label8,
        color='#8c6d31'
    )

    if drift_type == 'abrupt':
        plt.axvspan(
            drift_location,
            max(x_axis),
            color=Constants.color_yello,
            alpha=0.3,
            label='Concept Drift'
        )

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(
                x=loc,
                color=Constants.color_yello,
                linestyle='-',
                linewidth=.6,
                label=None
            )

        concept_drift_line = Line2D(
            [0], [0],
            color=Constants.color_yello,
            linestyle='-',
            linewidth=.6,
            label='Concept Drift'
        )

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(
            handles=[line1, line2, line3, line4, line5, line6, line7, line8, concept_drift_line],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.5,
            facecolor=Constants.color_light_blue,
            edgecolor=Constants.color_black
        )

    if drift_type == 'gradual':
        plt.legend(
            handles=[
                line1, line2, line3, line4, line5, line6, line7, line8,
                gradual_concept_drift_line[0], gradual_concept_drift_line[1]
            ],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if drift_type == 'abrupt':
        plt.legend(
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.5,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()


def plot_results_ten_models(
    x_axis,
    olr_wa_acc_list,
    olr_wa_mse_list,
    olr_wa_sccm_acc_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_acc_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_acc_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_adwin_sspt_r2_avg,
    olr_wa_adwin_sspt_mse_avg,
    olr_wa_adwin_ohl_r2_avg,
    olr_wa_adwin_ohl_mse_avg,
    olr_wa_kswin_reset_r2_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_r2_avg,
    olr_wa_kswin_window_mse_avg,
    olr_wa_kswin_sspt_r2_avg,
    olr_wa_kswin_sspt_mse_avg,
    olr_wa_kswin_ohl_r2_avg,
    olr_wa_kswin_ohl_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    label8,
    label9,
    label10,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts,
    save_path
):
    if kpi == 'R2':
        y_axis1 = olr_wa_acc_list
        y_axis2 = olr_wa_sccm_acc_list
        y_axis3 = olr_wa_adwin_reset_acc_list
        y_axis4 = olr_wa_adwin_window_acc_list
        y_axis5 = olr_wa_adwin_sspt_r2_avg
        y_axis6 = olr_wa_adwin_ohl_r2_avg
        y_axis7 = olr_wa_kswin_reset_r2_avg
        y_axis8 = olr_wa_kswin_window_r2_avg
        y_axis9 = olr_wa_kswin_sspt_r2_avg
        y_axis10 = olr_wa_kswin_ohl_r2_avg
    else:
        y_axis1 = olr_wa_mse_list
        y_axis2 = olr_wa_sccm_mse_list
        y_axis3 = olr_wa_adwin_reset_mse_list
        y_axis4 = olr_wa_adwin_window_mse_list
        y_axis5 = olr_wa_adwin_sspt_mse_avg
        y_axis6 = olr_wa_adwin_ohl_mse_avg
        y_axis7 = olr_wa_kswin_reset_mse_avg
        y_axis8 = olr_wa_kswin_window_mse_avg
        y_axis9 = olr_wa_kswin_sspt_mse_avg
        y_axis10 = olr_wa_kswin_ohl_mse_avg

    length = np.min([
        len(arr) for arr in [
            x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5,
            y_axis6, y_axis7, y_axis8, y_axis9, y_axis10
        ]
    ])

    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]
    y_axis3 = y_axis3[:length]
    y_axis4 = y_axis4[:length]
    y_axis5 = y_axis5[:length]
    y_axis6 = y_axis6[:length]
    y_axis7 = y_axis7[:length]
    y_axis8 = y_axis8[:length]
    y_axis9 = y_axis9[:length]
    y_axis10 = y_axis10[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)
        y_axis7 = np.log(y_axis7 + 1)
        y_axis8 = np.log(y_axis8 + 1)
        y_axis9 = np.log(y_axis9 + 1)
        y_axis10 = np.log(y_axis10 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(
        x_axis, y_axis1,
        linestyle='-',
        marker='o',
        markersize=1.5,
        linewidth=0.7,
        label=label1,
        color='#4c72b0'
    )

    line2, = plt.plot(
        x_axis, y_axis2,
        linestyle='-',
        marker='^',
        markersize=1.5,
        linewidth=1.0,
        label=label2,
        color='#dd8452'
    )

    line3, = plt.plot(
        x_axis, y_axis3,
        linestyle='--',
        marker='s',
        markersize=1.2,
        linewidth=0.7,
        label=label3,
        color='#55a868'
    )

    line4, = plt.plot(
        x_axis, y_axis4,
        linestyle=':',
        marker='D',
        markersize=1.2,
        linewidth=0.7,
        label=label4,
        color='#8172b3'
    )

    line5, = plt.plot(
        x_axis, y_axis5,
        linestyle=(0, (3, 1, 1, 1)),
        marker='P',
        markersize=1.3,
        linewidth=0.8,
        label=label5,
        color='#c44e52'
    )

    line6, = plt.plot(
        x_axis, y_axis6,
        linestyle=(0, (1, 1)),
        marker='*',
        markersize=2.0,
        linewidth=0.8,
        label=label6,
        color='#64b5cd'
    )

    line7, = plt.plot(
        x_axis, y_axis7,
        linestyle='-.',
        marker='v',
        markersize=1.3,
        linewidth=0.8,
        label=label7,
        color='#2a9d8f'
    )

    line8, = plt.plot(
        x_axis, y_axis8,
        linestyle=(0, (5, 1)),
        marker='X',
        markersize=1.3,
        linewidth=0.8,
        label=label8,
        color='#8c6d31'
    )

    line9, = plt.plot(
        x_axis, y_axis9,
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker='h',
        markersize=1.3,
        linewidth=0.8,
        label=label9,
        color='#e17c05'
    )

    line10, = plt.plot(
        x_axis, y_axis10,
        linestyle=(0, (7, 2)),
        marker='<',
        markersize=1.3,
        linewidth=0.8,
        label=label10,
        color='#76b7b2'
    )

    if drift_type == 'abrupt':
        plt.axvspan(
            drift_location,
            max(x_axis),
            color=Constants.color_yello,
            alpha=0.3,
            label='Concept Drift'
        )

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(
                x=loc,
                color=Constants.color_yello,
                linestyle='-',
                linewidth=.6,
                label=None
            )

        concept_drift_line = Line2D(
            [0], [0],
            color=Constants.color_yello,
            linestyle='-',
            linewidth=.6,
            label='Concept Drift'
        )

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(
            handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, concept_drift_line],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor=Constants.color_light_blue,
            edgecolor=Constants.color_black
        )

    if drift_type == 'gradual':
        plt.legend(
            handles=[
                line1, line2, line3, line4, line5,
                line6, line7, line8, line9, line10,
                gradual_concept_drift_line[0], gradual_concept_drift_line[1]
            ],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if drift_type == 'abrupt':
        plt.legend(
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()





def plot_results_ten_models_real_datasets(
    x_axis,
    olr_wa_acc_list,
    olr_wa_mse_list,
    olr_wa_sccm_acc_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_acc_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_acc_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_adwin_sspt_r2_avg,
    olr_wa_adwin_sspt_mse_avg,
    olr_wa_adwin_ohl_r2_avg,
    olr_wa_adwin_ohl_mse_avg,
    olr_wa_kswin_reset_r2_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_r2_avg,
    olr_wa_kswin_window_mse_avg,
    olr_wa_kswin_sspt_r2_avg,
    olr_wa_kswin_sspt_mse_avg,
    olr_wa_kswin_ohl_r2_avg,
    olr_wa_kswin_ohl_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    label8,
    label9,
    label10,
    log_enabled,
    legend_loc,
    save_path
):
    if kpi == 'R2':
        y_axis1 = olr_wa_acc_list
        y_axis2 = olr_wa_sccm_acc_list
        y_axis3 = olr_wa_adwin_reset_acc_list
        y_axis4 = olr_wa_adwin_window_acc_list
        y_axis5 = olr_wa_adwin_sspt_r2_avg
        y_axis6 = olr_wa_adwin_ohl_r2_avg
        y_axis7 = olr_wa_kswin_reset_r2_avg
        y_axis8 = olr_wa_kswin_window_r2_avg
        y_axis9 = olr_wa_kswin_sspt_r2_avg
        y_axis10 = olr_wa_kswin_ohl_r2_avg
    else:
        y_axis1 = olr_wa_mse_list
        y_axis2 = olr_wa_sccm_mse_list
        y_axis3 = olr_wa_adwin_reset_mse_list
        y_axis4 = olr_wa_adwin_window_mse_list
        y_axis5 = olr_wa_adwin_sspt_mse_avg
        y_axis6 = olr_wa_adwin_ohl_mse_avg
        y_axis7 = olr_wa_kswin_reset_mse_avg
        y_axis8 = olr_wa_kswin_window_mse_avg
        y_axis9 = olr_wa_kswin_sspt_mse_avg
        y_axis10 = olr_wa_kswin_ohl_mse_avg

    length = np.min([
        len(arr) for arr in [
            x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5,
            y_axis6, y_axis7, y_axis8, y_axis9, y_axis10
        ]
    ])

    x_axis = np.array(x_axis[:length])
    y_axis1 = np.array(y_axis1[:length])
    y_axis2 = np.array(y_axis2[:length])
    y_axis3 = np.array(y_axis3[:length])
    y_axis4 = np.array(y_axis4[:length])
    y_axis5 = np.array(y_axis5[:length])
    y_axis6 = np.array(y_axis6[:length])
    y_axis7 = np.array(y_axis7[:length])
    y_axis8 = np.array(y_axis8[:length])
    y_axis9 = np.array(y_axis9[:length])
    y_axis10 = np.array(y_axis10[:length])

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)
        y_axis7 = np.log(y_axis7 + 1)
        y_axis8 = np.log(y_axis8 + 1)
        y_axis9 = np.log(y_axis9 + 1)
        y_axis10 = np.log(y_axis10 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(x_axis, y_axis1, linestyle='-', marker='o',
                      markersize=1.5, linewidth=0.7, label=label1, color='#4c72b0')

    line2, = plt.plot(x_axis, y_axis2, linestyle='-', marker='^',
                      markersize=1.5, linewidth=1.0, label=label2, color='#dd8452')

    line3, = plt.plot(x_axis, y_axis3, linestyle='--', marker='s',
                      markersize=1.2, linewidth=0.7, label=label3, color='#55a868')

    line4, = plt.plot(x_axis, y_axis4, linestyle=':', marker='D',
                      markersize=1.2, linewidth=0.7, label=label4, color='#8172b3')

    line5, = plt.plot(x_axis, y_axis5, linestyle=(0, (3, 1, 1, 1)), marker='P',
                      markersize=1.3, linewidth=0.8, label=label5, color='#c44e52')

    line6, = plt.plot(x_axis, y_axis6, linestyle=(0, (1, 1)), marker='*',
                      markersize=2.0, linewidth=0.8, label=label6, color='#64b5cd')

    line7, = plt.plot(x_axis, y_axis7, linestyle='-.', marker='v',
                      markersize=1.3, linewidth=0.8, label=label7, color='#2a9d8f')

    line8, = plt.plot(x_axis, y_axis8, linestyle=(0, (5, 1)), marker='X',
                      markersize=1.3, linewidth=0.8, label=label8, color='#8c6d31')

    line9, = plt.plot(x_axis, y_axis9, linestyle=(0, (3, 1, 1, 1, 1, 1)), marker='h',
                      markersize=1.3, linewidth=0.8, label=label9, color='#e17c05')

    line10, = plt.plot(x_axis, y_axis10, linestyle=(0, (7, 2)), marker='<',
                       markersize=1.3, linewidth=0.8, label=label10, color='#76b7b2')

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    elif kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(True)
    sns.despine()

    plt.legend(
        handles=[
            line1, line2, line3, line4, line5,
            line6, line7, line8, line9, line10
        ],
        fontsize='small',
        loc=legend_loc,
        fancybox=True,
        shadow=True,
        borderpad=1,
        labelspacing=.1,
        facecolor='lightblue',
        edgecolor=Constants.color_black
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()


def plot_results_ten_models(
    x_axis,
    olr_wa_acc_list,
    olr_wa_mse_list,
    olr_wa_sccm_acc_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_acc_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_acc_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_adwin_sspt_r2_avg,
    olr_wa_adwin_sspt_mse_avg,
    olr_wa_adwin_ohl_r2_avg,
    olr_wa_adwin_ohl_mse_avg,
    olr_wa_kswin_reset_r2_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_r2_avg,
    olr_wa_kswin_window_mse_avg,
    olr_wa_kswin_sspt_r2_avg,
    olr_wa_kswin_sspt_mse_avg,
    olr_wa_kswin_ohl_r2_avg,
    olr_wa_kswin_ohl_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    label8,
    label9,
    label10,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts,
    save_path
):
    if kpi == 'R2':
        y_axis1 = olr_wa_acc_list
        y_axis2 = olr_wa_sccm_acc_list
        y_axis3 = olr_wa_adwin_reset_acc_list
        y_axis4 = olr_wa_adwin_window_acc_list
        y_axis5 = olr_wa_adwin_sspt_r2_avg
        y_axis6 = olr_wa_adwin_ohl_r2_avg
        y_axis7 = olr_wa_kswin_reset_r2_avg
        y_axis8 = olr_wa_kswin_window_r2_avg
        y_axis9 = olr_wa_kswin_sspt_r2_avg
        y_axis10 = olr_wa_kswin_ohl_r2_avg
    else:
        y_axis1 = olr_wa_mse_list
        y_axis2 = olr_wa_sccm_mse_list
        y_axis3 = olr_wa_adwin_reset_mse_list
        y_axis4 = olr_wa_adwin_window_mse_list
        y_axis5 = olr_wa_adwin_sspt_mse_avg
        y_axis6 = olr_wa_adwin_ohl_mse_avg
        y_axis7 = olr_wa_kswin_reset_mse_avg
        y_axis8 = olr_wa_kswin_window_mse_avg
        y_axis9 = olr_wa_kswin_sspt_mse_avg
        y_axis10 = olr_wa_kswin_ohl_mse_avg

    length = np.min([
        len(arr) for arr in [
            x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5,
            y_axis6, y_axis7, y_axis8, y_axis9, y_axis10
        ]
    ])

    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]
    y_axis3 = y_axis3[:length]
    y_axis4 = y_axis4[:length]
    y_axis5 = y_axis5[:length]
    y_axis6 = y_axis6[:length]
    y_axis7 = y_axis7[:length]
    y_axis8 = y_axis8[:length]
    y_axis9 = y_axis9[:length]
    y_axis10 = y_axis10[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)
        y_axis7 = np.log(y_axis7 + 1)
        y_axis8 = np.log(y_axis8 + 1)
        y_axis9 = np.log(y_axis9 + 1)
        y_axis10 = np.log(y_axis10 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(
        x_axis, y_axis1,
        linestyle='-',
        marker='o',
        markersize=1.5,
        linewidth=0.7,
        label=label1,
        color='#4c72b0'
    )

    line2, = plt.plot(
        x_axis, y_axis2,
        linestyle='-',
        marker='^',
        markersize=1.5,
        linewidth=1.0,
        label=label2,
        color='#dd8452'
    )

    line3, = plt.plot(
        x_axis, y_axis3,
        linestyle='--',
        marker='s',
        markersize=1.2,
        linewidth=0.7,
        label=label3,
        color='#55a868'
    )

    line4, = plt.plot(
        x_axis, y_axis4,
        linestyle=':',
        marker='D',
        markersize=1.2,
        linewidth=0.7,
        label=label4,
        color='#8172b3'
    )

    line5, = plt.plot(
        x_axis, y_axis5,
        linestyle=(0, (3, 1, 1, 1)),
        marker='P',
        markersize=1.3,
        linewidth=0.8,
        label=label5,
        color='#c44e52'
    )

    line6, = plt.plot(
        x_axis, y_axis6,
        linestyle=(0, (1, 1)),
        marker='*',
        markersize=2.0,
        linewidth=0.8,
        label=label6,
        color='#64b5cd'
    )

    line7, = plt.plot(
        x_axis, y_axis7,
        linestyle='-.',
        marker='v',
        markersize=1.3,
        linewidth=0.8,
        label=label7,
        color='#2a9d8f'
    )

    line8, = plt.plot(
        x_axis, y_axis8,
        linestyle=(0, (5, 1)),
        marker='X',
        markersize=1.3,
        linewidth=0.8,
        label=label8,
        color='#8c6d31'
    )

    line9, = plt.plot(
        x_axis, y_axis9,
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker='h',
        markersize=1.3,
        linewidth=0.8,
        label=label9,
        color='#e17c05'
    )

    line10, = plt.plot(
        x_axis, y_axis10,
        linestyle=(0, (7, 2)),
        marker='<',
        markersize=1.3,
        linewidth=0.8,
        label=label10,
        color='#76b7b2'
    )

    if drift_type == 'abrupt':
        plt.axvspan(
            drift_location,
            max(x_axis),
            color=Constants.color_yello,
            alpha=0.3,
            label='Concept Drift'
        )

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(
                x=loc,
                color=Constants.color_yello,
                linestyle='-',
                linewidth=.6,
                label=None
            )

        concept_drift_line = Line2D(
            [0], [0],
            color=Constants.color_yello,
            linestyle='-',
            linewidth=.6,
            label='Concept Drift'
        )

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(
            handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, concept_drift_line],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor=Constants.color_light_blue,
            edgecolor=Constants.color_black
        )

    if drift_type == 'gradual':
        plt.legend(
            handles=[
                line1, line2, line3, line4, line5,
                line6, line7, line8, line9, line10,
                gradual_concept_drift_line[0], gradual_concept_drift_line[1]
            ],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if drift_type == 'abrupt':
        plt.legend(
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()




def plot_results_ten_models_only_mse(
    x_axis,
    olr_wa_mse_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_adwin_sspt_mse_avg,
    olr_wa_adwin_ohl_mse_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_mse_avg,
    olr_wa_kswin_sspt_mse_avg,
    olr_wa_kswin_ohl_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    label8,
    label9,
    label10,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts,
    save_path
):

    y_axis1 = olr_wa_mse_list
    y_axis2 = olr_wa_sccm_mse_list
    y_axis3 = olr_wa_adwin_reset_mse_list
    y_axis4 = olr_wa_adwin_window_mse_list
    y_axis5 = olr_wa_adwin_sspt_mse_avg
    y_axis6 = olr_wa_adwin_ohl_mse_avg
    y_axis7 = olr_wa_kswin_reset_mse_avg
    y_axis8 = olr_wa_kswin_window_mse_avg
    y_axis9 = olr_wa_kswin_sspt_mse_avg
    y_axis10 = olr_wa_kswin_ohl_mse_avg

    length = np.min([
        len(arr) for arr in [
            x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5,
            y_axis6, y_axis7, y_axis8, y_axis9, y_axis10
        ]
    ])

    x_axis = x_axis[:length]
    y_axis1 = y_axis1[:length]
    y_axis2 = y_axis2[:length]
    y_axis3 = y_axis3[:length]
    y_axis4 = y_axis4[:length]
    y_axis5 = y_axis5[:length]
    y_axis6 = y_axis6[:length]
    y_axis7 = y_axis7[:length]
    y_axis8 = y_axis8[:length]
    y_axis9 = y_axis9[:length]
    y_axis10 = y_axis10[:length]

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)
        y_axis7 = np.log(y_axis7 + 1)
        y_axis8 = np.log(y_axis8 + 1)
        y_axis9 = np.log(y_axis9 + 1)
        y_axis10 = np.log(y_axis10 + 1)

    plt.figure(figsize=(10, 6))

    line1, = plt.plot(
        x_axis, y_axis1,
        linestyle='-',
        marker='o',
        markersize=1.5,
        linewidth=0.7,
        label=label1,
        color='#4c72b0'
    )

    line2, = plt.plot(
        x_axis, y_axis2,
        linestyle='-',
        marker='^',
        markersize=1.5,
        linewidth=1.0,
        label=label2,
        color='#dd8452'
    )

    line3, = plt.plot(
        x_axis, y_axis3,
        linestyle='--',
        marker='s',
        markersize=1.2,
        linewidth=0.7,
        label=label3,
        color='#55a868'
    )

    line4, = plt.plot(
        x_axis, y_axis4,
        linestyle=':',
        marker='D',
        markersize=1.2,
        linewidth=0.7,
        label=label4,
        color='#8172b3'
    )

    line5, = plt.plot(
        x_axis, y_axis5,
        linestyle=(0, (3, 1, 1, 1)),
        marker='P',
        markersize=1.3,
        linewidth=0.8,
        label=label5,
        color='#c44e52'
    )

    line6, = plt.plot(
        x_axis, y_axis6,
        linestyle=(0, (1, 1)),
        marker='*',
        markersize=2.0,
        linewidth=0.8,
        label=label6,
        color='#64b5cd'
    )

    line7, = plt.plot(
        x_axis, y_axis7,
        linestyle='-.',
        marker='v',
        markersize=1.3,
        linewidth=0.8,
        label=label7,
        color='#2a9d8f'
    )

    line8, = plt.plot(
        x_axis, y_axis8,
        linestyle=(0, (5, 1)),
        marker='X',
        markersize=1.3,
        linewidth=0.8,
        label=label8,
        color='#8c6d31'
    )

    line9, = plt.plot(
        x_axis, y_axis9,
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
        marker='h',
        markersize=1.3,
        linewidth=0.8,
        label=label9,
        color='#e17c05'
    )

    line10, = plt.plot(
        x_axis, y_axis10,
        linestyle=(0, (7, 2)),
        marker='<',
        markersize=1.3,
        linewidth=0.8,
        label=label10,
        color='#76b7b2'
    )

    if drift_type == 'abrupt':
        plt.axvspan(
            drift_location,
            max(x_axis),
            color=Constants.color_yello,
            alpha=0.3,
            label='Concept Drift'
        )

    if drift_type == 'incremental':
        for loc in range(drift_location, max(x_axis), drift_location):
            plt.axvline(
                x=loc,
                color=Constants.color_yello,
                linestyle='-',
                linewidth=.6,
                label=None
            )

        concept_drift_line = Line2D(
            [0], [0],
            color=Constants.color_yello,
            linestyle='-',
            linewidth=.6,
            label='Concept Drift'
        )

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()
        ]

    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2':
        plt.ylabel('R$^2$', fontsize=7)
    if kpi == 'MSE':
        plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine()
    plt.grid(True)

    if drift_type == 'incremental':
        plt.legend(
            handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, concept_drift_line],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor=Constants.color_light_blue,
            edgecolor=Constants.color_black
        )

    if drift_type == 'gradual':
        plt.legend(
            handles=[
                line1, line2, line3, line4, line5,
                line6, line7, line8, line9, line10,
                gradual_concept_drift_line[0], gradual_concept_drift_line[1]
            ],
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if drift_type == 'abrupt':
        plt.legend(
            fontsize='small',
            loc=legend_loc,
            fancybox=True,
            shadow=True,
            borderpad=1,
            labelspacing=.1,
            facecolor='lightblue',
            edgecolor=Constants.color_black
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()



def plot_results_ten_models_real_datasets_mse(
    x_axis,
    olr_wa_mse_list,
    olr_wa_sccm_mse_list,
    olr_wa_adwin_reset_mse_list,
    olr_wa_adwin_window_mse_list,
    olr_wa_adwin_sspt_mse_avg,
    olr_wa_adwin_ohl_mse_avg,
    olr_wa_kswin_reset_mse_avg,
    olr_wa_kswin_window_mse_avg,
    olr_wa_kswin_sspt_mse_avg,
    olr_wa_kswin_ohl_mse_avg,
    kpi,
    label1,
    label2,
    label3,
    label4,
    label5,
    label6,
    label7,
    label8,
    label9,
    label10,
    drift_location,
    log_enabled,
    legend_loc,
    drift_type,
    gradual_drift_locations,
    gradual_drift_concepts,
    save_path
):

    y_axis1 = olr_wa_mse_list
    y_axis2 = olr_wa_sccm_mse_list
    y_axis3 = olr_wa_adwin_reset_mse_list
    y_axis4 = olr_wa_adwin_window_mse_list
    y_axis5 = olr_wa_adwin_sspt_mse_avg
    y_axis6 = olr_wa_adwin_ohl_mse_avg
    y_axis7 = olr_wa_kswin_reset_mse_avg
    y_axis8 = olr_wa_kswin_window_mse_avg
    y_axis9 = olr_wa_kswin_sspt_mse_avg
    y_axis10 = olr_wa_kswin_ohl_mse_avg

    length = np.min([
        len(arr) for arr in [
            x_axis, y_axis1, y_axis2, y_axis3, y_axis4, y_axis5,
            y_axis6, y_axis7, y_axis8, y_axis9, y_axis10
        ]
    ])

    x_axis = np.array(x_axis[:length])
    y_axis1 = np.array(y_axis1[:length])
    y_axis2 = np.array(y_axis2[:length])
    y_axis3 = np.array(y_axis3[:length])
    y_axis4 = np.array(y_axis4[:length])
    y_axis5 = np.array(y_axis5[:length])
    y_axis6 = np.array(y_axis6[:length])
    y_axis7 = np.array(y_axis7[:length])
    y_axis8 = np.array(y_axis8[:length])
    y_axis9 = np.array(y_axis9[:length])
    y_axis10 = np.array(y_axis10[:length])

    if log_enabled:
        y_axis1 = np.log(y_axis1 + 1)
        y_axis2 = np.log(y_axis2 + 1)
        y_axis3 = np.log(y_axis3 + 1)
        y_axis4 = np.log(y_axis4 + 1)
        y_axis5 = np.log(y_axis5 + 1)
        y_axis6 = np.log(y_axis6 + 1)
        y_axis7 = np.log(y_axis7 + 1)
        y_axis8 = np.log(y_axis8 + 1)
        y_axis9 = np.log(y_axis9 + 1)
        y_axis10 = np.log(y_axis10 + 1)

    plt.figure(figsize=(10, 6))

    color_line1 = '#4c72b0'  # blue
    color_line2 = '#dd8452'  # orange
    color_line3 = '#228B22'  # forest green
    color_line4 = '#6A0DAD'  # purple
    color_line5 = '#2a9d8f'
    color_line6 = '#76b7b2'
    color_line7 = '#00A6D6'  # cyan blue
    color_line8 = '#8B4513'  # brown
    color_line9 = '#FF1493'  # deep pink
    color_line10 = '#FFD700'  # gold

    line1, = plt.plot(x_axis, y_axis1, linestyle='-', marker='o',
                      markersize=1.5, linewidth=0.7, label=label1, color=color_line1)

    line2, = plt.plot(x_axis, y_axis2, linestyle='-', marker='^',
                      markersize=1.5, linewidth=1.0, label=label2, color=color_line2)

    line3, = plt.plot(x_axis, y_axis3, linestyle='--', marker='s',
                      markersize=1.2, linewidth=0.7, label=label3, color=color_line3)

    line4, = plt.plot(x_axis, y_axis4, linestyle=':', marker='D',
                      markersize=1.2, linewidth=0.7, label=label4, color=color_line4)

    line5, = plt.plot(x_axis, y_axis5, linestyle=(0, (3, 1, 1, 1)), marker='P',
                      markersize=1.3, linewidth=0.8, label=label5, color=color_line5)

    line6, = plt.plot(x_axis, y_axis6, linestyle=(0, (1, 1)), marker='*',
                      markersize=2.0, linewidth=0.8, label=label6, color=color_line6)

    line7, = plt.plot(x_axis, y_axis7, linestyle='-.', marker='v',
                      markersize=1.3, linewidth=0.8, label=label7, color=color_line7)

    line8, = plt.plot(x_axis, y_axis8, linestyle=(0, (5, 1)), marker='X',
                      markersize=1.3, linewidth=0.8, label=label8, color=color_line8)

    line9, = plt.plot(x_axis, y_axis9, linestyle=(0, (3, 1, 1, 1, 1, 1)), marker='h',
                      markersize=1.3, linewidth=0.8, label=label9, color=color_line9)

    line10, = plt.plot(x_axis, y_axis10, linestyle=(0, (7, 2)), marker='<',
                       markersize=1.3, linewidth=0.8, label=label10, color=color_line10)

    plt.xlabel('$N$', fontsize=7)

    plt.ylabel('MSE', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(True)
    sns.despine()

    plt.legend(
        handles=[
            line1, line2, line3, line4, line5,
            line6, line7, line8, line9, line10
        ],
        fontsize='small',
        loc=legend_loc,
        fancybox=True,
        shadow=True,
        borderpad=1,
        labelspacing=.1,
        facecolor='lightblue',
        edgecolor=Constants.color_black
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {save_path}")

    plt.show()