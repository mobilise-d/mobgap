from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import Protocol, Self

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


class SklearnClassifier(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...  # noqa: N803

    def predict(self, X: ArrayLike) -> ArrayLike: ...  # noqa: N803


class SklearnScaler(Protocol):
    def fit(self, X: ArrayLike) -> Self: ...  # noqa: N803

    def transform(self, X: ArrayLike) -> ArrayLike: ...  # noqa: N803

    def fit_transform(self, X: ArrayLike) -> ArrayLike: ...  # noqa: N803

    def inverse_transform(self, X: ArrayLike) -> ArrayLike: ...  # noqa: N803
