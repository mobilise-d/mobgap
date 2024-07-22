(proj_struct)=
# Scope and Code Structure

To ensure that the project is easy to use, easy to maintain, and easy to expand in the future, all developers should
adhere to the guidelines outlined below.
Further the developers should familiarize themselves with aim and the scope of the project to make better decision when
it comes to including new functionality.
If you are new developer and need to familiarize yourself and need help with setting up your development environment to
work on this project, have a look at the [Development Guide](#dev_guide).

**Everything that follows are recommendations.**

As for every project you should:
- value your future self over your current self (don't use shortcuts or hacks that might become liabilities in the
long-term).
- think before you act.
- know the rules, before you break them.
- ask if in doubt.
- ask for a third opinion if you have two competing ones.

## Aim and Scope

The aim of the project is as follows:

1. Provide a **library** that provides that allows running the Mobilise-D pipeline as easy as possible on custom data
2. Provide a **toolbox** that allows users to build their own pipelines with the provided algorithms
3. Allow tooling for working with Mobilise-D data (raw IMU data and derived data) in general
4. Provide tooling for the evaluation of algorithm outputs
5. Grow into a general purpose library and toolbox for human gait and mobility analysis independent of the scope of the
   Mobilise-D project.

This means the library needs to require a high level interface for the core datasets and pipelines, but also act as 
flexible toolbox that can be the bases for general research in the field.
For this we should try to follow the following principles:

1. Classes and functions should only require the bare minimum of input they require
2. Variations in functionality should be made available to users either by providing appropriate keyword arguments,
   separate methods, or even separate functions or classes.
3. Following 2., each function/class should have one function and one function only.
   In the case where "one function" consist of multiple steps (e.g. event detection consists of IC-detection, 
   and TO-detection) and it makes sense to group them together to provide an easier interface, the individual 
   functions should still be available to users to allow them to skip or modify steps as they desire.
4. The library should not prevent users from "intentional stupidity".
   For example, if users decide to apply an event detection method designed to only run on data from sensors attached to 
   the lower-back on data from a wrist-worn sensor, **let them**.
5. Whenever possible the library should allow to provide native datatypes as outputs.
   Native datatypes in this case includes all the container objects Python supports (lists, dicts, etc.) and the base 
   datatypes of `numpy` and `pandas` (np.array, pd.DataFrame, pd.Series). Only if it improves usability and adds 
   significant value (either for understanding or manipulating the output), custom datatypes should be used.
   One example of that would be the `tpcp.Dataset` datatype used for high level pipelines.
6. The library should be agnostic to sensor systems and should not contain any code that is highly specific to a certain
   IMU system. This means that loading and preprocessing should be handled by the user or other libraries.
   However, mobgap can provide utils to make preprocessing easier for users.


## Code Structure

### Library Structure

As the library aims to support multiple algorithms, each algorithm with similar function should be grouped into 
individual modules/folders (e.g. gait sequence detection, event detection, ...).
Each folder contains a `base.py` file that contains the base classes for the respective type of algorithm.
The actual algorithms are implemented in separate "hidden" files (i.e. leading `_`) in the respective folder and
the public functions and classes are imported in the `__init__.py` of the folder.
In the `__init__` files, the `__all__` list should be used to allow for `*` imports without cluttering the 
namespace.
If an algorithm requires large amount of code and multiple classes/functions, it can be refactored into its own
submodule.

### Import Paths

In general, we use absolute imports internally (instead of relative imports), even though it requires more typing.

.. warning:: With absolute imports and the use of `__init__` imports, you can sometimes create circular dependencies.
   E.g. when you want to import a function `a()` defined in `mod/_a.py` in `mod/_b.py` and `mod/_b.py` and `mod/_a.py`
   are both imported in `mod/__init__.py` (as by convention explained above), you must import `c()` with its full 
   internal path `from mod._c import c` instead of `from mod import c` to avoid circular imports.

In case a single function from an external package is used, just import this function.
In case multiple functions from an external package are used, import this package/module under a commonly used alias
(e.g. `np` for numpy, `pd` for pandas, ...)

### Algorithms with specific dependencies

Some algorithms require specific dependencies, that are not required by anything else.
In this case, they should be made optional dependencies of mobgap to reduce the number of dependencies for users.
To make this possible, the algorithm should be implemented in a separate submodule, that is not imported anywhere else 
in the library.

In case, where only a few methods from a different library are required, it might also be feasible to just copy the
required methods into the submodule and add a comment to the source of the code.
Make sure to check the license of the original code and provide sufficient attribution.
For example, we are doing this with some functions from the `gaitmap` library.

### Helper Functions and Utils

Functions that can be reused across multiple algorithms of similar type should be placed in a module level `_utils.py` 
file (e.g. `gsd/_utils.py`). Functions that are reusable across multiple modules should be placed in an
appropriate file in the package level `utils` module (e.g. `utils/math_helper.py`).
Utils are generally "private", but when you can imagine usecases where users might reasonably need to use them when 
working with the library, you can expose them in the respective `__init__` files.

### Class Structure

All larger algorithms should be represented by classes and not by functions for a couple of reasons that are explained 
below.
See the [general guides in tpcp](https://tpcp.readthedocs.io/en/latest/guides/algorithms_pipelines_datasets.html).

From the guide:

- The `__init__` of each class should only be there to set parameters. No parameter validation or any functionality 
  should be performed here.
- No actual data should be passed to the `__init__`. Think of it as configuring the algorithm.
- Defaults for **all** parameters should be provided in the `__init__`.
- The names of the class attributes should be identical to the parameter names used in the `__init__`
  (i.e. the init should contain statements like: `self.parameter_name = parameter_name`).
- All parameters that are not directly depended on the input data, should be set in the `__init__` and not in the
  *action* method (see below).
  This also includes parameters that should be adapted based on the data, but can theoretically estimated without having
  the data available.
  All these parameters should be set in the `__init__`.
  The data and all other measured/directly data-depended parameters are passed in the action method.
  This includes for example, the raw IMU data, the sampling rate demographic information, etc..
- Results and outputs are stored with a trailing underscore (e.g. `filtered_stride_list_`).
- All algorithms of the same type should have a consistent interface with (as far as possible), identical input 
  parameters to allow drop-in replacements of different algorithms
- Each type of algorithm has one (or multiple) "action" methods with a descriptive name.
  These *action* methods take the actual data as input and will produce results.
- All *action* methods just return `self` (the object itself)

Additions to the guide:

- All algorithms classes must directly or indirectly inherit from `tpcp.Algorithm`
- All classes should store the data (and other arguments) passed in the "action" step in the class object unless the 
  amount of data would result in an unreasonable performance issue.
  Ideally this should be a reference and not a copy of the data! This allows to path the final object as a whole to 
  helper functions, that e.g. can visualize in and outputs (see also this [Q&A entry](#q&a__other_paras))
  These parameters should be documented under "Other Parameters" to not clutter the docstring.
- Mutable defaults in the init are as always a bad idea, but in mobgap we make specific exceptions.
  See [this guide in tpcp](https://tpcp.readthedocs.io/en/latest/guides/general_concepts.html#mutable-defaults).
- All methods should take care that they do not modify the original data passed to the function.
  If required a copy of the data can be created, but **not** stored in the object.
- All classes should validate their input parameters during the "action" (or whenever the parameters are first needed).
  Don't overdue the validation and focus on logical validation (e.g. a value can not be larger than x) and not on type 
  validation.
  For type validation, we should trust that Python provides the correct error message once an invalid step is performed.
- All classes should inherent from a BaseClass specific to their type that implements common functionality and enforces 
  the interface. Remember to call respective `super` methods when required.
  The resulting class structure should look like this:

```
tpcp.Algorithm -> Basic setting of parameters
|
Base<AlgorithmType> -> Basic interface to ensure all algos of the same type use the same input and outputs for their 
|                      action methods
|
<TheActualAlgorithm>
|
<VariationsOfAlgorithm> -> A improved version of an algorithm, when it does not make sense to toggle the improvement 
                           via an inputparameter on the algorithm
```


#### Example class structure

Below you can find the simplified class structure of the `RamppEventDetection` algorithm.
It should serve as an example on how further algorithms should be implemented and documented.
Note, the use of the `base_icd_docfiller`, which is used to interpolate common sections of the docstring.

Also review the actual implementation of the other algorithms for further inspiration and guidance.

```python
from typing import Any

import pandas as pd
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap.data_transform import EpflDedriftedGaitFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.initial_contacts.base import BaseIcDetector, base_icd_docfiller


@base_icd_docfiller
class IcdIonescu(BaseIcDetector):
    """Implementation of the initial_contacts algorithm by McCamley et al. (2012) [1]_ modified by Ionescu et al. (2020) [2]_.

    The algorithm includes the following steps starting from vertical acceleration
    of the lower-back during a gait sequence:

    1. Resampling: 100 Hz --> 40 Hz
    2. Band-pass filtering --> lower cut-off: 0.15 Hz; higher cut-off: 3.14 Hz
    3. Cumulative integral --> cumulative trapezoidal integration
    4. Continuous Wavelet Transform (CWT) --> Ricker wavelet
    5. Zero crossings detection
    6. Detect peaks between zero crossings --> negative peaks = ICs

    This is based on the implementation published as part of the mobilised project [3]_.
    However, this implementation deviates from the original implementation in some places.
    For details, see the notes section and the examples.

    Parameters
    ----------
    pre_filter
        A pre-processing filter to apply to the data before the initial_contacts algorithm is applied.
    cwt_width
        The width of the wavelet

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(ic_list_)s

    Notes
    -----
    Points of deviation from the original implementation and their reasons:

    - We use a different downsampling method, which should be "more" correct from a signal theory perspective,
      but will yield slightly different results.
    - We use a slightly different approach when it comes to the detection of the peaks between the zero crossings.
      However, the results of this step are identical to the matlab implementation.

    .. [1] J. McCamley, M. Donati, E. Grimpampi, C. Mazzà, "An enhanced estimate of initial contact and final contact
       instants of time using lower trunk inertial sensor data", Gait & Posture, vol. 36, no. 2, pp. 316-318, 2012.
    .. [2] A. Paraschiv-Ionescu, A. Soltani and K. Aminian, "Real-world speed estimation using single trunk IMU:
       methodological challenges for impaired gait patterns," 2020 42nd Annual International Conference of the IEEE
       Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada, 2020, pp. 4596-4599,
       doi: 10.1109/EMBC44109.2020.9176281.
    .. [3] https://github.com/mobilise-d/Mobilise-D-TVS-Recommended-Algorithms/blob/master/ICDA/Library/SD_algo_AMC.m

    """

    pre_filter: BaseFilter
    cwt_width: float

    # Some constants of the algorithms
    _INTERNAL_FILTER_SAMPLING_RATE_HZ: int = 40

    def __init__(self, *, pre_filter: BaseFilter = cf(EpflDedriftedGaitFilter()), cwt_width: float = 9.0) -> None:
        self.pre_filter = pre_filter
        self.cwt_width = cwt_width

    @base_icd_docfiller
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_short)s.

        %(detect_info)s

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s

        """
        # Setting the Other Parameters
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Here is where the actual Algo implementation lives
        ...
        # Setting teh results
        self.ic_list_ = ...

        return self
```

### Random and Initial State

If any algorithms rely on random processes/operations, the random state should be configurable, by an optional kwarg in
the `__init__` called `random_state`.
We follow the [`sklearn` recommendations](https://scikit-learn.org/stable/glossary.html#term-random_state) on this.

Algorithms that require an initial value for some optimization should expose this value via the `__init__`.
If the parameter is `None` a random initial value should be used that is controlled by the additional `random_state`
argument.

## Code guidelines

All code should follow coherent best practices.
As far as possible the adherence to these best practices should be tested using linters or a testsuite that runs as part
of the CI.

### General Codestyle

For general codestyle we follow [PEP8](https://www.python.org/dev/peps/pep-0008/) with a couple of exceptions
(e.g. line length).
These are documented in the linter config (`.ruff.toml`)

### Naming

We follow the naming conventions outlined in [PEP8](https://www.python.org/dev/peps/pep-0008/#naming-conventions).

For algorithms (if no better name is available) we use `AlgotypeAuthorName` (e.g. `IcdIonescu`).

### Documentation

For documentation, we follow [numpys guidelines](https://numpydoc.readthedocs.io/en/latest/format.html).
If the datatype is already provided as TypeHint (see below) it does not need to be specified in the docstring again.
However, it might be helpful to document additional type information (e.g. the shape of an array that can not be
captured by the TypeHint)

All user-facing functions (all functions and methods that do not have a leading underscore) are expected to be properly
and fully documented for potential users.
All private functions are expected to be documented in a way that other developer can understand them.
Additionally, each module should have a docstring explaining its content.
If a module contains only one class this can a single sentence/word (e.g. `"""Event detection based on ... ."""`).

### Typehints

To provide a better developer experience the library should use
[TypeHints](https://numpydoc.readthedocs.io/en/latest/format.html) where ever possible.

Remember to use `np.ndarray` instead of `np.array` as type specification of numpy arrays.
