import uuid

import numpy as np
import pandas as pd
import pytest


def window(start, end, **parameter):
    parameter = parameter or {}

    parameter = {**parameter, "duration": end - start}

    return dict(s_id=str(uuid.uuid4()), start=start, end=end, **parameter)


@pytest.fixture
def naive_stride_list():
    """A window list full of identical strides."""
    x = np.arange(0, 10100, 100)
    start_end = zip(x[:-1], x[1:])

    return pd.DataFrame.from_records(
        [window(start=s, end=e, para_1=1) for i, (s, e) in enumerate(start_end)]
    ).set_index("s_id")
