Formatting Pandas Tables
========================

.. automodule:: mobgap.utils.tables
    :no-members:
    :no-inherited-members:

Format Transformers
-------------------
Formatters and helper classes that will change the value in Dataframes with the goal of making them
stylistic more pleasing.
These are more complex transforms compared to what is possible via the `pandas` styling API.

.. currentmodule:: mobgap.utils.tables

.. autosummary::
   :toctree: ../generated/utils/tables
   :template: class.rst

   FormatTransformer


Stylers
-------
Functions that effect the styler of a dataframe.
The helpers create functions that can be applied to the Styler object of a dataframe for various effects.

.. currentmodule:: mobgap.utils.tables
.. autosummary::
   :toctree: ../generated/utils/tables
   :template: function.rst

    best_in_group_styler
    border_after_group_styler
    compare_to_threshold_styler
    revalidation_table_styles


Types
-----
.. currentmodule:: mobgap.utils.tables
.. autosummary::
   :toctree: ../generated/utils/tables
   :template: class.rst

    ValueWithMetadata
    CustomFormattedValueWithMetadata
    RevalidationInfo

