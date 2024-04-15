from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import Protocol, Self

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


class SklearnClassifier(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...

    def predict(self, X: ArrayLike) -> ArrayLike: ...


class SklearnScaler(Protocol):
    def fit(self, X: ArrayLike) -> Self: ...

    def transform(self, X: ArrayLike) -> ArrayLike: ...

    def fit_transform(self, X: ArrayLike) -> ArrayLike: ...

    def inverse_transform(self, X: ArrayLike) -> ArrayLike: ...
