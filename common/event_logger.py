import csv
import datetime
import logging
import os
from enum import Enum, auto
from pathlib import Path


class LogEventType(Enum):
    REGISTRATION = auto(),
    MODEL_UPDATE = auto(),
    VIOLATION = auto(),
    HORIZON_UPDATE = auto(),
    MEASUREMENT = auto(),
    PREDICTION = auto(),
    MODEL_TRAIN = auto()


class LogEvent(object):
    def __init__(self, node_id: str, event: LogEventType, message: str = '-') -> None:
        self._node_id = node_id
        self._event = event
        self._message = message

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def event(self) -> LogEventType:
        return self._event

    @property
    def message(self) -> str:
        return self._message


class EventLogger(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logging.debug("Creating new instance of EventLogger")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, 'events.csv')
        header = ['timestamp', 'node_id', 'event', 'message']
        with open(path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            self._path = path

    def log_event(self, event: LogEvent) -> None:
        entry = [datetime.datetime.now(), event.node_id, event.event.name, event.message]
        with open(self._path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(entry)
