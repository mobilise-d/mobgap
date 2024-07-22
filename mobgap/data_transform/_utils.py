from typing import Any

from typing_extensions import Unpack

from mobgap.data_transform.base import BaseTransformer
from mobgap.utils.dtypes import DfLikeT


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
    for name, transformer in transformers:
        try:
            transformer_with_results = transformer.transform(data, **kwargs)
            data = transformer_with_results.transformed_data_
        except Exception as e:
            raise RuntimeError(
                f"Error while applying transformer '{name}' in the transformer chain. "
                "Scroll up to see the full traceback of this error."
            ) from e
        # We ask the transformer that was just use to potentially update the kwargs for the next transformer
        kwargs = transformer_with_results._get_updated_chain_kwargs(**kwargs)
    return data
