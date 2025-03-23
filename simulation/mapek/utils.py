import json
import os
import shutil
from tracemalloc import Snapshot

import matplotlib.dates as mdates
import pandas as pd
from matplotlib import axes, pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from sensor.sensor_knowledge import SensorKnowledge


def plot_simulation(path: str,
                    sim_id: str,
                    predictions: pd.DataFrame,
                    measurements: pd.DataFrame,
                    violations: pd.DataFrame,
                    horizon_updates: pd.DataFrame,
                    configuration_updates: pd.DataFrame,
                    last_config: pd.Series,
                    threshold: float,
                    ):
    fig, plots = plt.subplots(figsize=(200, 10))
    plots: axes.Axes

    # Measurements, predictions
    plots.plot(predictions.index, predictions['TL'], label='Predictions', zorder=6, color='darkorange')
    plots.plot(measurements.index, measurements['TL'], label='Measurements', zorder=5, color='black')

    # Violations
    nearest_indices = predictions.index.get_indexer(violations.index, method='nearest')
    violation_predictions = predictions.iloc[nearest_indices]
    plots.scatter(violation_predictions.index, violation_predictions['TL'], label='Violation', s=12, zorder=8,
                  color='red', marker='x')

    # Horizon updates
    nearest_indices = predictions.index.get_indexer(horizon_updates.index, method='nearest')
    horizon_updates_predictions = predictions.iloc[nearest_indices]
    plots.scatter(horizon_updates_predictions.index,
                  horizon_updates_predictions['TL'], label='Horizon Update', s=12, zorder=8, color='limegreen',
                  marker='8')

    # Threshold
    plots.plot(measurements.index, measurements['TL'] + threshold, label='_Measurements (upper threshold)',
               color='gray', linestyle='dashed')
    plots.plot(measurements.index, measurements['TL'] - threshold, label='_Measurements (lower threshold)',
               color='gray', linestyle='dashed')
    plt.fill_between(
        measurements.index, measurements['TL'] - threshold, measurements['TL'] + threshold,
        color='gray', alpha=0.2, label=f'Threshold (±{threshold}°C)'
    )

    # Configuration updates
    patches = []
    new_labels = []
    plotted_configurations = []
    colors = [
        '#FF9999', '#66B2FF', '#7FFF00', '#FFD700', '#DA70D6', '#FFA500', '#FF69B4', '#1E90FF', '#8A2BE2', '#32CD32'
    ]
    configuration_colors = {}
    configuration_numbers = {}
    colors_counter = 0

    configs = configuration_updates.copy()
    configs.loc[measurements.index.min()] = configuration_updates.iloc[0] if last_config is None else last_config
    configs.loc[measurements.index.max()] = configuration_updates.iloc[-1] if len(
        configuration_updates) > 0 else configs.iloc[-1]
    configs.sort_index(inplace=True)
    for (timestamp, configuration), (next_timestamp, _) in zip(configs.iterrows(),
                                                               configs[1:].iterrows()):
        config_id = configuration[0]
        if config_id not in plotted_configurations:
            configuration_colors[config_id] = colors[colors_counter]
            config_number = colors_counter + 1
            configuration_numbers[config_id] = config_number
            colors_counter += 1
            patch = Patch(edgecolor='black', facecolor=configuration_colors[config_id], label=config_number)
            patches.append(patch)
            new_labels.append(f"Active Model [{config_number}]: {config_id}")

        plotted_configurations.append(config_id)
        plots.axvspan(timestamp, next_timestamp,
                      color=configuration_colors[config_id],
                      alpha=0.15,
                      label=f"_{config_id}")

    plots.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plots.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plots.yaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(rotation=90)
    plots.margins(x=0.01, y=0.01)

    plots.grid(True)
    plots.set_title('Simulation')
    plots.set_ylabel('Temperature (°C)')
    handles, labels = plots.get_legend_handles_labels()
    plt.legend(handles=handles + patches, labels=labels + new_labels, loc='best')
    file_path = os.path.join(path, f"{sim_id}.png")
    fig.savefig(file_path, format='png', bbox_inches='tight')


def save_results(
        bs_models_dir: str,
        sim_dir: str,
        knowledge: SensorKnowledge,
        profiler_data: pd.DataFrame,
        sim_data: pd.DataFrame,
        threshold: float,
        memory_snapshot: Snapshot,
        cpu_time: float
) -> None:
    os.makedirs(sim_dir, exist_ok=True)

    source = os.path.join(bs_models_dir)
    dest = os.path.join(sim_dir, bs_models_dir)
    shutil.copytree(source, dest, dirs_exist_ok=True)

    violations = knowledge.predictor.data.get_violations()
    violations.to_csv(os.path.join(sim_dir, "violations.csv"))

    measurements = knowledge.predictor.data.get_measurements()
    measurements = measurements[measurements.index >= sim_data.index.min()]
    measurements.to_csv(os.path.join(sim_dir, "measurements.csv"))

    predictions = knowledge.predictor.data.get_predictions()
    predictions.to_csv(os.path.join(sim_dir, "predictions.csv"))

    configuration_updates = knowledge.predictor.data.get_configuration_updates()
    configuration_updates.index = pd.to_datetime(configuration_updates.index)
    configuration_updates.to_csv(os.path.join(sim_dir, "config_updates.csv"))

    horizon_updates = knowledge.predictor.data.get_horizon_updates()
    horizon_updates.to_csv(os.path.join(sim_dir, "horizon_updates.csv"))

    analysis = knowledge.predictor.data.get_analysis()
    analysis.to_csv(os.path.join(sim_dir, "analysis.csv"))

    interaction_log = profiler_data[
        (profiler_data['func'] == 'send_update') |
        (profiler_data['func'] == 'send_violation') |
        (profiler_data['func'] == 'sync')
        ]

    profiler_data.to_csv(os.path.join(sim_dir, "profiling.csv"))
    stats = profiler_data.drop(columns=["timestamp", "details"])
    model_deployments = knowledge.predictor.data.get_model_deployments()
    model_deployments.to_csv(os.path.join(sim_dir, "model_deployments.csv"))
    stats_by_func = stats.groupby("func")
    stats_sum = stats_by_func.sum()
    stats_sum['calls'] = stats_by_func.size().values
    stats_sum.to_csv(os.path.join(sim_dir, "profiling_sum.csv"))
    stats_avg = stats_by_func.mean()
    stats_avg.to_csv(os.path.join(sim_dir, "profiling_avg.csv"))

    statistics = memory_snapshot.statistics("filename")
    peak_memory = 0
    total_allocations = 0
    memory_data = []
    for stat in statistics:
        _size = stat.size / (1024 * 1024)
        peak_memory += _size
        total_allocations += stat.count
        memory_data.append({
            "traceback": stat.traceback,
            "size_megabytes": _size,
            "count": stat.count,
        })

    pd.DataFrame(memory_data).to_csv(os.path.join(sim_dir, "memory_snapshot.csv"))

    aggregate = stats.drop(columns='func').sum()

    results = {
        "measurements": len(measurements),
        "violations": len(violations),
        "violations_sent_to_BS": len(interaction_log[interaction_log['func'] == 'send_violation']),
        "analysis": len(analysis),
        "configuration_updates": len(configuration_updates),
        "horizon_updates": len(horizon_updates),
        "horizon_updates_sent_to_BS": len(interaction_log[interaction_log['func'] == 'send_update']),
        "sync_requests_to_BS": len(interaction_log[interaction_log['func'] == 'sync']),
        "model_deployments": len(model_deployments),
        "total_cpu_time_s": round(cpu_time, 3),
        "total_memory_allocations": total_allocations,
        "peak_memory_allocated_MB": round(peak_memory, 3),
        "total_data_received_B": aggregate["received_data_B"],
        "total_data_sent_B": aggregate["transmitted_data_B"],
        "total_data_exchanged_B": aggregate["received_data_B"] + aggregate["transmitted_data_B"]
    }

    results_file_path = os.path.join(sim_dir, "results.json")
    with open(results_file_path, "w") as fp:
        json.dump(results, fp)

    last_config = configuration_updates.loc[configuration_updates.index.min()]
    min_date = measurements.index.min()
    max_date = measurements.index.max()
    window_size = pd.DateOffset(months=3)

    start_date: pd.Timestamp = min_date

    while start_date <= max_date:
        end_date = start_date + window_size
        configs = configuration_updates.loc[start_date:end_date]
        sim_id = f"{start_date.year}_{start_date.month:02d}_{start_date.day:02d}"

        plot_simulation(
            sim_dir, sim_id,
            predictions[start_date:end_date],
            measurements[start_date:end_date],
            violations[start_date:end_date],
            horizon_updates[start_date:end_date],
            configs,
            last_config,
            threshold
        )
        if len(configs) > 0:
            last_config = configs.iloc[-1]
        start_date += window_size
