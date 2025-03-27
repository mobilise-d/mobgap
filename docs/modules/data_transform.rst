Data Transformations and Filters
================================

.. automodule:: mobgap.data_transform
    :no-members:
    :no-inherited-members:

Filter
++++++
.. currentmodule:: mobgap.data_transform

.. autosummary::
   :toctree: generated/data_transform
   :template: class.rst

    ButterworthFilter
    FirFilter
    CwtFilter
    SavgolFilter
    GaussianFilter
    EpflGaitFilter
    EpflDedriftFilter
    EpflDedriftedGaitFilter

Transformations
+++++++++++++++
.. currentmodule:: mobgap.data_transform

.. autosummary::
   :toctree: generated/data_transform
   :template: class.rst

    Resample
    Pad
    Crop


Utilities
+++++++++

.. currentmodule:: mobgap.data_transform

.. autosummary::
   :toctree: generated/data_transform
   :template: function.rst

   chain_transformers

Base Classes
++++++++++++
.. automodule:: mobgap.data_transform.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.data_transform.base

.. autosummary::
   :toctree: generated/data_transform
   :template: class.rst

    BaseTransformer
    BaseFilter
    FixedFilter
    ScipyFilter
    IdentityFilter

Docu-helper
-----------

.. autosummary::
   :toctree: generated/data_transform
   :template: func.rst

    base_filter_docfiller
    fixed_filter_docfiller
    scipy_filter_docfiller