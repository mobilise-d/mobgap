import contextlib
import time
from collections.abc import Generator, Iterable
from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypedDict, TypeVar

from mobgap._docutils import make_filldoc

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc


class MeasureTimeResults(TypedDict):
    """Results of the measure_time context manager."""

    start_datetime_utc_timestamp: float
    start_datetime: str
    end_datetime_utc_timestamp: float
    end_datetime: str
    runtime_s: float


class Timer:
    results: MeasureTimeResults

    def start(self) -> None:
        self.reset()
        self.results = {
            "start_datetime_utc_timestamp": datetime.now(UTC).timestamp(),
            "start_datetime": datetime.now().astimezone().isoformat(),
            "perf_counter_start": time.perf_counter(),
        }

    def stop(self) -> None:
        self.results["end_datetime_utc_timestamp"] = datetime.now(UTC).timestamp()
        self.results["end_datetime"] = datetime.now().astimezone().isoformat()
        self.results["runtime_s"] = time.perf_counter() - self.results["perf_counter_start"]
        self.results.pop("perf_counter_start")

    def reset(self) -> None:
        self.results = {}


@contextlib.contextmanager
def measure_time() -> Generator[MeasureTimeResults, None, None]:
    """Context manager to measure the execution time.

    Note: This is not meant for high precision timing.
    """
    timer = Timer()
    timer.start()
    yield timer.results
    timer.stop()


C = TypeVar("C", bound=Callable)


def timed_action_method(meth: C) -> C:
    """Measure the execution time of an action method.

    Note: This is not meant for high precision timing.
    Results are stored on the instance in the `perf_` attribute.
    """

    @wraps(meth)
    def timed_method(self, *args, **kwargs):  # noqa: ANN202, ANN001, ANN002, ANN003
        with measure_time() as timer:
            result = meth(self, *args, **kwargs)
        self.perf_ = timer
        return result

    return timed_method


timer_doc_filler = make_filldoc(
    {
        "perf_": """
        perf_
            A dictionary with the performance results of the action method.
            This includes:

            - start_datetime_utc_timestamp: The start time of the action in UTC as a timestamp.
            - start_datetime: The start time of the action as a string.
            - end_datetime_utc_timestamp: The end time of the action in UTC as a timestamp.
            - end_datetime: The end time of the action as a string.
            - runtime_s: The runtime of the action in seconds.
        """
    }
)


def set_attrs_from_dict(obj: Any, attr_dict: dict[str, Any], *, key_postfix: str = "", key_prefix: str = "") -> None:
    """Set attributes of an object from a dictionary.

    Parameters
    ----------
    obj
        The object to set the attributes on.
    attr_dict
        The dictionary with the attributes to set.
    key_postfix
        A postfix to add to the key before setting the attribute.
        For example set it to "_" to add an underscore after each key and mark the attributes as results in the tpcp
        algorithms.
    key_prefix
        A prefix to add to the key before setting the attribute.
        For example set it to "_" to add an underscore before each key and mark the attributes as private.
    """
    for key, value in attr_dict.items():
        setattr(obj, f"{key_prefix}{key}{key_postfix}", value)


def invert_list_of_dicts(list_of_dicts: Iterable[dict[str, any]]) -> dict[str, list[any]]:
    """Invert a list of dictionaries.

    Parameters
    ----------
    list_of_dicts
        The list of dictionaries to invert.

    Returns
    -------
    dict
        The inverted dictionary.
    """
    inverted_dict = {}
    for key in list_of_dicts[0]:
        inverted_dict[key] = [d[key] for d in list_of_dicts]
    return inverted_dict
