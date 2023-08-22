from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# plot graph of metrics
def plot_graphs(metrics_df, gt_df=None, fig_size=(16,8)):
    first_set = list(metrics_df.keys())[0]

    metrics = list(metrics_df[first_set].keys())
    n_metrics = len(metrics)

    fig, axs = plt.subplots(1, n_metrics, figsize=fig_size)
    colors = [color['color'] for i_color, color in enumerate(plt.rcParams['axes.prop_cycle']) 
              if i_color < len(metrics_df.keys())]

    for i_set, set_type in enumerate(metrics_df.keys()):
        set_color = colors[i_set]
        for i, metric in enumerate(metrics):
            axs[i].plot(metrics_df[set_type][metric], label=set_type, color=set_color)
            if gt_df is not None and metric in gt_df[set_type].keys():
                gt_val = gt_df[set_type][metric]
                gt_val = gt_val[0] if isinstance(gt_val, list) else gt_val[1]
                axs[i].axhline(y=gt_val, linestyle='--', label=f'{set_type}_ASR', color=set_color)

            axs[i].set_title(metric)
            axs[i].legend()

    plt.show()


if __name__ == '__main__':
    metrics = pd.DataFrame({'train': {'loss': [1, 2, 3], 'acc': [0.1, 0.2, 0.3]},
                            'dev': {'loss': [2, 3, 4], 'acc': [0.2, 0.3, 0.4]}})
    
    gt_df = pd.DataFrame({'train': {'acc': [0.01]},
                            'dev': {'acc': [0.02]}})
    plot_graphs(metrics, gt_df)

    # metrics_df = {
    #     'train': pd.read_csv(r'results\DebugRecAce\2023-08-22_23-18-56\train_metrics.csv', index_col=0),
    #     'dev': pd.read_csv(r'results\DebugRecAce\2023-08-22_23-18-56\dev_metrics.csv', index_col=0)
    # }

    # gt_df = {
    #     'train': pd.read_csv(r'results\ASR\train_metrics.csv', index_col=0),
    #     'dev': pd.read_csv(r'results\ASR\dev_metrics.csv', index_col=0)
    # }

    # plot_graphs(metrics_df, gt_df=gt_df, fig_size=(20,8))