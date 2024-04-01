from typing import Optional, List

import seaborn as sns
from matplotlib import pyplot as plt

from base.window_generator import WindowGenerator
from common import DataStorage


def plot_history(history, metric: str, path: Optional[str] = None, show: bool = True) -> None:
    loss = history.history[metric]
    val_loss = history.history[f'val_{metric}']
    epochs = range(1, len(loss) + 1)
    fig: plt.Figure = plt.figure()
    plt.plot(epochs, loss, 'y', label="Training")
    plt.plot(epochs, val_loss, 'b', label="Validation")
    plt.title(metric)
    plt.legend()
    if show:
        plt.show()
    if path is not None:
        fig.savefig(path)
    plt.close(fig)


def plot_distribution(window_generator: WindowGenerator, columns: Optional[List[str]] = None):
    if columns is None:
        columns = window_generator.df.columns
    features = window_generator.df[columns]
    df = features.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df)
    _ = ax.set_xticklabels(features.keys(), rotation=90)
    plt.show()


def plot(window_generator: WindowGenerator, plot_col: str, model=None, max_subplots=3, title=None):
    inputs, outputs = window_generator.example
    plt.figure(figsize=(12, 8))
    plot_col_index = window_generator.input_features_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(window_generator.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if window_generator.output_features:
            output_col_index = window_generator.output_features_indices.get(plot_col, None)
        else:
            output_col_index = plot_col_index

        if output_col_index is None:
            continue

        plt.scatter(window_generator.output_indices, outputs[n, :, output_col_index],
                    edgecolors='k', label='Outputs', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(window_generator.output_indices, predictions[n, :, output_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    if title:
        plt.suptitle(title)

    plt.xlabel('Time [h]')
    plt.show()


def plot(data_storage: DataStorage):
    """Creates a plot for every attribute, comparing measurements and predictions."""
    for col in data_storage._measurements.columns:
        plt.plot(data_storage._measurements[col], label='Measurement')
        plt.plot(data_storage._predictions[col], label='Prediction')
        plt.show()
