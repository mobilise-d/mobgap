"""
Pretrained models and predefined Parameters
===========================================

All mobgap algorithms are implemented as tpcp classes (see :class:`tpcp.Algorithm`).
Hence, all there configuration is stored in the parameters of the class.

Each algorithm already has sensible default parameters, which can be used for many applications.
However, for some algorithms we have some known good parameters for specific use cases.
For example, for the usage in specific clinical cohorts.
Similarly, the machine learning based algorithms have a set of pretrained models that we provide with the package.

For both cases, we need a good way to make these parameters and models available to the user.
Below, we show the agreed upon way to do this.
For reference about the discussion, see `GitHub Issue <https://github.com/mobilise-d/mobgap/issues/65>`_.

The basic idea is to add a static inner class to the algorithm class, which contains the parameters and models.
In the easiest case, these are just dictionaries.
If more complex configuration is needed, we can use the ``property`` decorator to load these parameters from a file.
In both cases we wrap the returned dictionaries in ``MappingProxyType`` to make them readonly, to make sure the user
does not accidentally change the predefined parameters using the ``update`` method of the dict.

The simple case
---------------
For a real world implementation see :class:`mobgap.wba.WbAssembly`.

For the simple case we just add a static inner class to the algorithm class.
For the example we assume that the algorithm has two parameters ``param1`` and ``param2`` and we assume that we have
two sets of predefined parameters for healthy and pathological gait.

We will omit the algorithm implementation (aka the action method) here, as it is not relevant for the example.
"""

from types import MappingProxyType
from typing import Optional

import tpcp


class MyAlgorithm(tpcp.Algorithm):
    param1: float
    param2: str

    def __init__(self, param1: float = 2, param2: str = "foo"):
        self.param1 = param1
        self.param2 = param2

    class PredefinedParameters:
        healthy = MappingProxyType({"param1": 1, "param2": "healthy"})
        pathological = MappingProxyType({"param1": 3, "param2": "pathological"})


# %%
# Now we can use the predefined parameters as follows:
healthy_params = MyAlgorithm.PredefinedParameters.healthy

algo_with_healthy_params = MyAlgorithm(**healthy_params)
algo_with_healthy_params.get_params()

# %%
# This way, we can easily store a set of different parameters for different use cases.
# Users are still able to overwrite the parameters, if they want to.
# For example, only use the predefined parameters for some parameters and overwrite others:
pathological_params = MyAlgorithm.PredefinedParameters.pathological
algo_with_custom_pathological_params = MyAlgorithm(
    **dict(pathological_params, param2="bar")
)
algo_with_custom_pathological_params.get_params()

# %%
# Other possible syntax versions (based on preference):
pathological_params = MyAlgorithm.PredefinedParameters.pathological
algo_with_custom_pathological_params = MyAlgorithm(
    **{**pathological_params, "param2": "bar"}
)
algo_with_custom_pathological_params.get_params()

# %%
# Or
pathological_params = MyAlgorithm.PredefinedParameters.pathological
algo_with_custom_pathological_params = MyAlgorithm(
    **(pathological_params | dict(param2="bar"))
)
algo_with_custom_pathological_params.get_params()

# %%
# Or by using set params:
algo_with_custom_pathological_params = MyAlgorithm(**pathological_params)
algo_with_custom_pathological_params.set_params(param2="bar")
algo_with_custom_pathological_params.get_params()


# %%
# Depending on the specific case, we can also use one of the predefined parameters as default values for the
# constructor.
# We can use the :func:`tpcp.misc.set_defaults` function to do this easily.
from tpcp.misc import set_defaults


class MyAlgorithm(tpcp.Algorithm):
    param1: float
    param2: str

    class PredefinedParameters:
        healthy = MappingProxyType({"param1": 1, "param2": "healthy"})
        pathological = MappingProxyType({"param1": 3, "param2": "pathological"})

    @set_defaults(**PredefinedParameters.healthy)
    def __init__(
        self,
        param1: float,
        param2: str,
    ):
        self.param1 = param1
        self.param2 = param2


algo = MyAlgorithm()
algo.get_params()

# %%
# Loading parameters from a file (or other source)
# ------------------------------------------------
# Sometimes, we have a lot of parameters, that we don't want to hardcode in the source code, or we want to include
# objects in the parameters that can not be easily hardcoded (e.g. a trained machine learning model).
#
# In this case, we can use the :class:`~tpcp.misc.classproperty`` decorator to load the parameters from a file
# (or any other source).
# As our parameter class is "just" a class, we can easily add such a property to it.
#
# Let's assume our algorithm has an additional parameter ``model`` that takes an optional pre-trained ML model.
# We will provide different versions of predefined parameters, that use different models.
#
# For the example, we will not actually load a model, but just use a string.
from tpcp.misc import classproperty


class MyAlgorithm(tpcp.Algorithm):
    param1: float
    param2: str
    model: Optional[str]

    class PredefinedParameters:
        @classmethod
        def _load_from_file(cls, model_name):
            # Load the model from a file here
            # We could even add a cashing mechanism here, if we want to avoid loading the model multiple times.
            print(f"Loading model {model_name} from file")
            return "model_" + model_name

        @classproperty
        def healthy(cls):
            # Load the model from a file here
            model = cls._load_from_file("healthy")
            return MappingProxyType(
                {"param1": 1, "param2": "healthy", "model": model}
            )

        @classproperty
        def pathological(cls):
            # Load the model from a file here
            model = cls._load_from_file("pathological")
            return MappingProxyType(
                {"param1": 3, "param2": "pathological", "model": model}
            )

    def __init__(
        self,
        param1: float = 2,
        param2: str = "foo",
        model: Optional[str] = None,
    ):
        self.param1 = param1
        self.param2 = param2
        self.model = model


# %%
# Now we can use the predefined parameters as before, but the file loading is only done when we actually use the
# parameters, and we avoid loading all the models into memory at the beginning.

healthy_params = MyAlgorithm.PredefinedParameters.healthy
healthy_params

# %%
algo_with_healthy_params = MyAlgorithm(**healthy_params)
algo_with_healthy_params.get_params()
