from matplotlib import pyplot as plt


# plot graph of loss and WER
def plot_graphs(metrics):
    fig, axs = plt.subplots(2, 1, figsize=(8, 14))
    axs[0].plot(metrics['loss']['train'], label='train')
    axs[0].plot(metrics['loss']['dev'], label='dev')
    axs[0].set_title('Loss')
    axs[0].legend()

    axs[1].plot(metrics['wer']['train'], label='train')
    axs[1].plot(metrics['wer']['dev'], label='dev')
    axs[1].set_title('WER')
    axs[1].legend()

    plt.show()


if __name__ == '__main__':
    metrics = {'loss': {'train': [1, 2, 3], 'dev': [4, 5, 6]}, 'wer': {'train': [0.1, 0.2, 0.3], 'dev': [0.4, 0.5, 0.6]}}
    plot_graphs(metrics)