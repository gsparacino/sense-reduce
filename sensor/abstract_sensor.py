import datetime
from abc import ABC, abstractmethod

import pandas as pd


class AbstractSensor(ABC):
    @property
    @abstractmethod
    def measurement(self) -> (datetime.datetime, pd.Series):
        """Returns a list of measurements indexed by their name."""
        pass

    @abstractmethod
    def can_read_measurements(self) -> bool:
        pass
