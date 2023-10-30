Data Transformations and Filters
================================

.. automodule:: gaitlink.data_transform
    :no-members:
    :no-inherited-members:

Filter
++++++
.. currentmodule:: gaitlink.data_transform

.. autosummary::
   :toctree: generated/data_transform
   :template: class.rst

    ButterworthFilter
    FirFilter
    EpflGaitFilter
    EpflDedriftFilter
    EpflDedriftedGaitFilter

Utilities
+++++++++

.. currentmodule:: gaitlink.data_transform

.. autosummary::
   :toctree: generated/data_transform
   :template: function.rst

   chain_transformers

Base Classes
++++++++++++
.. automodule:: gaitlink.data_transform.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: gaitlink.data_transform.base

.. autosummary::
   :toctree: generated/data_transform
   :template: class.rst

    BaseTransformer
    BaseFilter
    FixedFilter
    ScipyFilter

Docu-helper
-----------

.. autosummary::
   :toctree: generated/data_transform
   :template: func.rst

    base_filter_docfiller
    fixed_filter_docfiller
    scipy_filter_docfiller