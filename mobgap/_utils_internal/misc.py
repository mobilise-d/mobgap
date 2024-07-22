import contextlib
import time
from collections.abc import Generator
from datetime import datetime
from typing import Any, TypedDict


class MeasureTimeResults(TypedDict):
    """Results of the measure_time context manager."""

    start_datetime_utc_timestamp: float
    start_datetime: str
    end_start_datetime_utc_timestamp: float
    end_start_datetime: str
    runtime: float


@contextlib.contextmanager
def measure_time() -> Generator[MeasureTimeResults, None, None]:
    """Context manager to measure the execution time.

    Note: This is not meant for high precision timing.
    """
    results = {
        "start_datetime_utc_timestamp": datetime.utcnow().timestamp(),
        "start_datetime": datetime.now().astimezone().isoformat(),
    }
    start_time = time.perf_counter()
    yield results
    end_time = time.perf_counter()
    results["end_datetime_utc_timestamp"] = datetime.utcnow().timestamp()
    results["end_datetime"] = datetime.now().astimezone().isoformat()
    results["runtime_s"] = end_time - start_time


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
