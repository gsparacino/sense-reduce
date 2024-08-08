import datetime
import logging
from abc import ABC, abstractmethod
from typing import Optional


class ViolationsMonitor(ABC):
    """
    A utility class that keeps track of recent violations and checks if the number of violations exceeds the configured
    limit.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the counter to its initial state.
        """
        pass

    @abstractmethod
    def add_violation(self, timestamp: datetime.datetime) -> bool:
        """
        Add a violation to the monitor and check if the new violation brings the counter above the threshold.

        Args:
            timestamp: the timestamp of the violation

        Returns: True if the new violation brings the counter above the threshold, false otherwise
        """
        pass

    @abstractmethod
    def is_above_threshold(self) -> bool:
        """
        Checks whether the current violations in the monitor put its counter above the threshold.

        Returns: True if the monitor's violations are above the threshold, False otherwise
        """
        pass

    @abstractmethod
    def get_violations_count(self) -> int:
        """
        Returns: the number of violations currently stored in the monitor
        """
        pass

    @abstractmethod
    def get_violations_limit(self) -> int:
        """
        Returns: the maximum number of violations that can be added to the monitor in a given time span before the
        counter's threshold is exceeded.
        """
        pass

    @abstractmethod
    def get_latest_violation_timestamp(self) -> Optional[datetime.datetime]:
        """
        Returns the timestamp of the most recent violation, or None if there are no violations stored in the monitor.
        """
        pass

    @abstractmethod
    def get_first_violation_timestamp(self) -> Optional[datetime.datetime]:
        """
        Returns the timestamp of the oldest violation, or None if there are no violations stored in the monitor.
        """
        pass


class TimeWindowViolationsMonitor(ViolationsMonitor):
    def __init__(self, violation_time_threshold: datetime.timedelta, violation_count_threshold: int):
        """
        Counts the violations that occurred within the provided time span.
        A counter violation occurs when the number of violations in the counter exceeds the configured value
        (violation_count_threshold).
        A violation is automatically removed from the counter when its timestamp is less than
        (now - violation_time_threshold).

        Args:
            violation_time_threshold: the time limit to set as threshold for violations
            violation_count_threshold: the maximum number of violations that can occur within the given time threshold
        """
        self._violations: list[datetime.datetime] = []
        self.violation_time_threshold = violation_time_threshold
        self.violation_count_threshold = violation_count_threshold

    def reset(self) -> None:
        self._violations = []

    def add_violation(self, timestamp: datetime.datetime) -> bool:
        self._delete_old_violations()
        self._violations.append(timestamp)
        logging.debug(
            f"New violation ({len(self._violations)}/{self.violation_count_threshold})"
        )
        return self.is_above_threshold()

    def is_above_threshold(self) -> bool:
        return len(self._violations) >= self.violation_count_threshold

    def get_violations_count(self) -> int:
        return len(self._violations)

    def get_violations_limit(self) -> int:
        return self.violation_count_threshold

    def get_latest_violation_timestamp(self) -> datetime.datetime:
        if len(self._violations) > 0:
            return self._violations[-1]
        else:
            return datetime.datetime.now()

    def get_first_violation_timestamp(self) -> datetime.datetime:
        if len(self._violations) > 0:
            return self._violations[0]
        else:
            return datetime.datetime.now() - self.violation_time_threshold

    def _delete_old_violations(self) -> None:
        now = datetime.datetime.now()
        for violation in self._violations:
            cutoff = now - self.violation_time_threshold
            if violation < cutoff:
                logging.debug(f"Removing violation from ViolationMonitor: {violation} is older than cutoff {cutoff}")
                self._violations.remove(violation)
