Data loading and Datasets
=========================

.. automodule:: gaitlink.data
    :no-members:
    :no-inherited-members:

General Gait Data
-----------------

Base Classes
++++++++++++
.. currentmodule:: gaitlink.data.base

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    BaseGaitDataset
    BaseGaitDatasetWithReference

Generic Loader Classes
++++++++++++++++++++++
.. currentmodule:: gaitlink.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    GaitDatasetFromData


Mobilise-D Matlab format
------------------------

Base Classes
++++++++++++
.. currentmodule:: gaitlink.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    BaseGenericMobilisedDataset
    GenericMobilisedDataset

Load Functions
++++++++++++++

.. autosummary::
   :toctree: generated/data
   :template: function.rst

    load_mobilised_matlab_format
    parse_reference_parameters


Example Data
------------

Dataset Classes
+++++++++++++++
.. currentmodule:: gaitlink.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    LabExampleDataset

Functional Interface
++++++++++++++++++++

Load Functions
++++++++++++++

.. autosummary::
   :toctree: generated/data
   :template: function.rst

    get_all_lab_example_data_paths

Datatypes
---------
.. currentmodule:: gaitlink.data.base

.. autosummary::
   :toctree: generated/data
   :template: namedtuple.rst

    ReferenceData

.. currentmodule:: gaitlink.data

.. autosummary::
   :toctree: generated/data
   :template: namedtuple.rst

    MobilisedTestData
    MobilisedMetadata

Docfiller
---------
.. currentmodule:: gaitlink.data.base

.. autosummary::
   :toctree: generated/data
   :template: function.rst

    base_gait_dataset_docfiller

.. currentmodule:: gaitlink.data

.. autosummary::
   :toctree: generated/data
   :template: function.rst

    matlab_dataset_docfiller
