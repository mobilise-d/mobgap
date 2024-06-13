"""
.. _generic_data_transforms:

Generic Data Transforms
=======================

Besides explicit gait specific algorithms, we also provide a set of generic data transforms, which can be used as part
of an algorithm or to pre-process the data before applying a gait specific algorithm.

Examples for such generic data transforms are:

- Resampling
- Filtering
- ...

This example provides some high level information about how to use these generic data transforms and implement your
own.
If you want to learn how to work with specific data transforms, please refer to the respective example.

Why a more complicated interface?
---------------------------------

Many of the transforms we provide as algorithm classes could be perfectly covered by a simple function.
Frankly, many of them actually are already implemented as functions in other scientific python packages (e.g. scipy).
However, wrapping them again by a custom code object has a couple of advantages:

- We can add input and output data handling specific to our datatypes
- We can change defaults and how the method is called, in particular if we use a method for something that it was not
  intended for (e.g. using cwt as a filter)
- Adding our own docstrings and examples that explain the usage in the specific context of gait analysis

In addition, wrapping them in custom classes based on the :class:`tpcp.Algorithm` class allows us to separate
configuration (parameters passed to the `__init__` method) an execution of the transform (calling the `transform`
method).
This unifies the calling interface of all transforms and allows us to use them in a pipeline, further it allows to pass
around transformation objects as arguments to other methods and algorithms (see example below).

Creating your own data transform
--------------------------------

To understand how generic data transforms work, we will create a simple example transform, that we can then use for
further examples.
We will create two transforms: One that adds a constant and one that multiplies the data by a factor.

The basic things that a data-transform needs are:

- Inherit from :class:`~mobgap.data_transform.base.BaseTransformer` (or one of its subclasses, e.g.
  :class:`~mobgap.data_transform.base.BaseFilter`)
- Implement the ``transform`` method
- The transform method should support data-frames, series, and numpy arrays as input and output.
  The output should match the input function.
  :func:`~mobgap.utils.dtypes.dflike_as_2d_array` can be used for that.

"""

from typing import Any

import numpy as np
import pandas as pd
from mobgap.data_transform.base import BaseTransformer
from mobgap.utils.dtypes import DfLike, dflike_as_2d_array
from tpcp import Algorithm
from typing_extensions import Self, Unpack


class ShiftTransformer(BaseTransformer):
    """A simple data transform that shifts the data by a constant."""

    shift_by: float

    def __init__(self, shift_by: float = 0) -> None:
        self.shift_by = shift_by

    def transform(self, data: DfLike, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Shift the data by a constant.

        Parameters
        ----------
        data
            The data to shift.
            Must be dataframe-like (i.e. a dataframe, series, or numpy array).

        Returns
        -------
        self
            The instance of the class with the ``transformed_data_`` attribute set to the shifted data.

        """
        # The use of `dflike_as_2d_array` is not strictly necessary here, as all inputs would support the addition
        # operation.
        # But we use it here to show the general structure.
        data_array, index, to_dflike = dflike_as_2d_array(data)
        data_array = data_array + self.shift_by
        self.transformed_data_ = to_dflike(data_array, index)
        return self


class ScaleTransformer(BaseTransformer):
    """A simple data transform that scales the data by a constant."""

    scale_by: float

    def __init__(self, scale_by: float = 1) -> None:
        self.scale_by = scale_by

    def transform(self, data: DfLike, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Scale the data by a constant.

        Parameters
        ----------
        data
            The data to scale.
            Must be dataframe-like (i.e. a dataframe, series, or numpy array).

        Returns
        -------
        self
            The instance of the class with the ``transformed_data_`` attribute set to the scaled data.

        """
        data_array, index, to_dflike = dflike_as_2d_array(data)
        data_array = data_array * self.scale_by
        self.transformed_data_ = to_dflike(data_array, index)
        return self


# %%
# Using a data transform
# ----------------------
# To apply a data transform to some data, first create an instance of the transform and then call the ``transform``
# method with the data.
# The ``transform`` method returns the instance of the transform with the ``transformed_data_`` attribute set to the
# transformed data.
#
# Below we show this with some very basic example data:

data = np.array([1, 2, 3])
data

# %%
scale_transformer = ScaleTransformer(scale_by=2)
scale_transformer.transform(data)
scale_transformer.transformed_data_

# %%
shift_transformer = ShiftTransformer(shift_by=1)
shift_transformer.transform(data)
shift_transformer.transformed_data_

# %%
# Note, that we could reuse the same instance of the transform to apply it to different data.
# In this case, it makes sense to call ``clone()`` on the instance before calling ``transform`` to ensure that all data
# related attributes are reset.
new_data = np.array([4, 5, 6])

cloned_transformer = scale_transformer.clone().transform(new_data)
cloned_transformer.transformed_data_

# %%
# Chaining data transforms
# ------------------------
# As all transformers have the same interface, we can chain them together using
# :func:`~mobgap.data_transform.chain_transformers`.
#
# TODO: Update once we have a chain_transformer class
from mobgap.data_transform import chain_transformers

chained_result = chain_transformers(
    data, [("scale", scale_transformer), ("shift", shift_transformer)]
)
chained_result

# %%
# Transformer as arguments to other algorithms
# --------------------------------------------
# As all transformers have the same interface, we can pass them as arguments to other algorithms or pipelines.
# This is not unique to transformer classes, but true for all tpcp algorithm objects.
#
# To demonstrate this, we create a simple algorithm that takes a transformer as an argument and applies it to the data.
from tpcp import cf


class MyComplicatedAlgorithm(Algorithm):
    pre_processing: BaseTransformer

    result_: pd.DataFrame

    def __init__(
        self, pre_processing: BaseTransformer = cf(ShiftTransformer(shift_by=1))
    ) -> None:
        self.pre_processing = pre_processing

    def run(self, data: pd.DataFrame) -> Self:
        pre_processed_data = (
            self.pre_processing.clone().transform(data).transformed_data_
        )

        # Here we would do something more complicated with the data
        # For now we skip this and just return the data on the result attribute
        self.result_ = pre_processed_data

        return self


# %%
# This algorithm now uses a transformer as "pre-processing".
# We specify a default value, but the user could pass any transformer instance they want.
# This makes it really easy expose all possible parameters and configurations of a transformer to the user.
#
# Below a couple of ways of how the algorithm can be used:
#
# Using the default value:
my_algorithm = MyComplicatedAlgorithm()
my_algorithm.run(pd.DataFrame(data)).result_

# %%
# Supplying a different transformer:
my_algorithm = MyComplicatedAlgorithm(
    pre_processing=ScaleTransformer(scale_by=2)
)
my_algorithm.run(pd.DataFrame(data)).result_

# %%
# Modifying the transformer after creating the algorithm:
my_algorithm = MyComplicatedAlgorithm()
my_algorithm.set_params(pre_processing=ScaleTransformer(scale_by=3))
my_algorithm.run(pd.DataFrame(data)).result_

# %%
# Modifying nested parameters of the transformer after creating the algorithm:
my_algorithm = MyComplicatedAlgorithm()
my_algorithm.set_params(pre_processing__shift_by=-1)
my_algorithm.run(pd.DataFrame(data)).result_

# %%
# Some final notes:
#
# - The ``set_params`` method is inherited from :class:`tpcp.Algorithm` and is available for all algorithms.
#   It supports nested parameters, i.e. you can use ``__`` to specify parameters of nested objects.
# - To pass a series of transformers to an algorithm, you can use tpcp's composite parameters (see the example
#   `here <https://tpcp.readthedocs.io/en/latest/auto_examples/recipies/_03_composite_objects.html>`__).
# - When using an instance as a default value, you should wrap it in the :func:`~tpcp.cf` function.
#   This will ensure, that a new instance is created for each call of the algorithm.
#   Learn more about this
#   `here <https://tpcp.readthedocs.io/en/latest/guides/general_concepts.html#mutable-defaults>`__.
