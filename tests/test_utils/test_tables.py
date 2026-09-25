import numpy as np
import pandas as pd

from mobgap.utils.tables import StatsFunctions


def test_pairwise_tests_returns_p_value_for_reference_comparison():
    data = pd.DataFrame({"group": ["reference"] * 5 + ["other"] * 5, "value": [1, 2, 3, 4, 5, 2, 3, 4, 5, 6]})

    comparisons = StatsFunctions.pairwise_tests(data, "value", "group", "reference")

    assert np.isfinite(comparisons["other"]["p"])
