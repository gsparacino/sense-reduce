import csv
import datetime
import functools
import logging
import os
import time
import tracemalloc
from typing import Optional

import pandas as pd
import psutil


class Profiler:
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            logging.debug("Creating new instance of Profiler")
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, log_dir: os.path):
        self.properties = [
            'timestamp',
            'func',
            'details',
            'elapsed_time_ms',
            'cpu_time_ms',
            'peak_memory_usage_MB',
            'received_data_B',
            'transmitted_data_B'
        ]
        path = os.path.join(log_dir, 'profiling.csv')
        self._log_path = path
        self.data = pd.DataFrame(columns=self.properties)

    def to_csv(self):
        self._init_log_file()
        self.data.to_csv(self._log_path)

    def _init_log_file(self):
        logging.debug(f"Saving computation log to {self._log_path}")
        with open(self._log_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.properties)
            writer.writeheader()

    def profile(self, func, details: str, args, kwargs):
        """
        Profiles a function by recording its execution time, cpu time and memory usage.

        :param func: the function to profile
        :param details: additional details about the profiled function
        :param args: the arguments of the function to profile
        :param kwargs: the keyword arguments of the function to profile
        :return: the result of the execution of the profiled function
        """
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
        self.add_log_entry(
            timestamp,
            end_time - start_time,
            func.__name__,
            details,
            peak_memory_usage,
            total_cpu_time,
            0,
            0
        )
        return result

    def add_log_entry(
            self,
            timestamp: datetime.datetime,
            execution_time: float,
            func_name: str,
            details: Optional[str],
            peak_memory_usage: int,
            total_cpu_time: float,
            received_data: int,
            transmitted_data: int
    ):
        """
        Adds a new entry to the "profile log", which monitors the computational resources utilization of the profiled
        functions.

        :param timestamp: the timestamp of the entry, as a datetime object
        :param execution_time: the "wall-clock time" for the execution of the profiled function, in milliseconds
        :param func_name: the name of the profiled function
        :param details: some additional information about the profiled function
        :param peak_memory_usage: the maximum memory allocated by the profiled function, in bytes
        :param total_cpu_time: the total "cpu time" for the execution of the profiled function, in milliseconds
        :param received_data: the total size of data received from other devices via HTTP, in bytes
        :param transmitted_data: the total size of data transmitted to other devices via HTTP, in bytes
        """
        entry = [
            timestamp,
            func_name,
            details,
            round(execution_time * 1000, 2),
            round(total_cpu_time * 1000, 2),
            round(peak_memory_usage / (1024 ** 2), 2),
            received_data,
            transmitted_data
        ]
        # with open(self._log_path, 'a', newline='') as log:
        #     appender = csv.writer(log)
        #     appender.writerow(entry)
        self.data.loc[len(self.data)] = entry

    def reset_data(self):
        self.data = pd.DataFrame(columns=self.properties)


_profiler: Optional[Profiler] = None


def init_profiler(log_dir: os.path) -> None:
    global _profiler
    _profiler = Profiler(log_dir)


def get_profiler() -> Optional[Profiler]:
    global _profiler
    if _profiler is not None:
        return _profiler


def profiled(tag: str = None):
    """
    Annotates a function with a profiling decorator, which gathers data on the computational resources utilization of
    the profiled function.

    :param tag: the ID of the function, as it will appear in the "profile log". If None, the name of the function
        (func.__name__) is used.
    """

    def inner(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _profiler is not None:
                return _profiler.profile(func, tag, args, kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return inner
