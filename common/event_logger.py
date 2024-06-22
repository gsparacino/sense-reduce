import csv
import datetime
import logging
import os
from enum import Enum, auto
from pathlib import Path


class LogEventType(Enum):
    REGISTRATION = auto(),
    MODEL_SEND = auto(),
    MODEL_SWITCH = auto(),
    VIOLATION = auto(),
    PORTFOLIO_SYNC = auto(),
    MEASUREMENT = auto(),
    MODEL_TRAIN = auto(),
    MODEL_READY = auto()


class LogEvent(object):
    def __init__(self, node_id: str, event: LogEventType, details: str = '-') -> None:
        self._node_id = node_id
        self._event = event
        self._details = details

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def event(self) -> LogEventType:
        return self._event

    @property
    def details(self) -> str:
        return self._details


class EventLogger(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logging.debug("Creating new instance of EventLogger")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: str = None):
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            path = os.path.join(path, 'events.csv')
            header = ['timestamp', 'node_id', 'event', 'details']
            with open(path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                self.log_file = path

    def log_event(self, event: LogEvent) -> None:
        if self.log_file is not None:
            entry = [datetime.datetime.now(), event.node_id, event.event.name, event.details]
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(entry)
