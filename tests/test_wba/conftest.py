import uuid

import numpy as np
import pytest


def window(start, end, parameter=None, **kwargs):
    parameter = parameter or {}
    default_parameter = {
        "length": 1,
    }

    parameter = {**default_parameter, **parameter}
    return dict(id=str(uuid.uuid4()), start=start, end=end, **kwargs, parameter=parameter)


@pytest.fixture()
def naive_stride_list():
    """A window list full of identical strides."""
    x = np.arange(0, 10100, 100)
    start_end = zip(x[:-1], x[1:])

    return [window(start=s, end=e) for i, (s, e) in enumerate(start_end)]


@pytest.fixture()
def naive_event_list():
    dummy_events = [
        {
            "name": "event1",
            "events": [window(i, i + 20, {"angle": 20}) for i in range(55, 8000, 1000)],
        },
        {"name": "event2", "events": [window(1500, 4000), window(6500, 7000)]},
    ]

    return dummy_events
