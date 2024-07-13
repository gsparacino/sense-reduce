import csv
import datetime
import functools
import os
import time
import tracemalloc

import psutil

profile_log_path = os.getenv('PROFILE_LOG_PATH')

if profile_log_path is not None:
    with open(profile_log_path, mode='w', newline='') as file:
        properties = [
            'timestamp', 'func', 'elapsed_time_ms',
            'cpu_time_ms',
            'peak_memory_usage_MB'
        ]
        writer = csv.DictWriter(file, fieldnames=properties)
        writer.writeheader()

    file = open(profile_log_path, 'a', newline='')


def profiled(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if profile_log_path is None:
            return func(*args, **kwargs)

        process = psutil.Process(os.getpid())

        start_cpu_times = process.cpu_times()
        start_time = time.time()
        timestamp = datetime.datetime.now()
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak_memory_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        end_time = time.time()
        end_cpu_times = process.cpu_times()

        total_cpu_time = (
                end_cpu_times.user - start_cpu_times.user +
                end_cpu_times.system - start_cpu_times.system +
                end_cpu_times.children_user - start_cpu_times.children_user +
                end_cpu_times.children_system - start_cpu_times.children_system
        )

        entry = [
            timestamp,
            func.__name__,
            (end_time - start_time) * 1000,
            total_cpu_time * 1000,
            peak_memory_usage / (1024 ** 2)
        ]
        appender = csv.writer(file)
        appender.writerow(entry)

        return result

    return wrapper
