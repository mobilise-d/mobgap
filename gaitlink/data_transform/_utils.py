from typing import Any

from typing_extensions import Unpack

from gaitlink.data_transform.base import BaseTransformer
from gaitlink.utils.dtypes import DfLikeT


def chain_transformers(
    data: DfLikeT, transformers: list[tuple[str, BaseTransformer]], **kwargs: Unpack[dict[str, Any]]
) -> DfLikeT:
    """Chain multiple transformers together.

    Parameters
    ----------
    data
        The data to be transformed.
    transformers
        A list of tuples, where the first element is the name of the transformer and the second element is the
        transformer instance itself.
    kwargs
        Further keyword arguments for the transform function.

    Returns
    -------
    data
        The transformed data.

    """
    # TODO: At the moment, we don't have any way to handle transformers that change kwargs somehow.
    #       For example the resampler would need to change the sampling_rate_hz in the kwargs for all subsequent steps.
    for name, transformer in transformers:
        try:
            data = transformer.clone().transform(data, **kwargs).transformed_data_
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Error while applying transformer '{name}' in the transformer chain. "
                "Scroll up to see the full traceback of this error."
            ) from e
    return data
