Data loading and data management
================================

.. automodule:: gaitlink.data
    :no-members:
    :no-inherited-members:

Mobilise-D Matlab format
------------------------

Dataset Classes
+++++++++++++++
.. currentmodule:: gaitlink.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    GenericMobilisedDataset

Load Functions
++++++++++++++

.. autosummary::
   :toctree: generated/data
   :template: function.rst

    load_mobilised_matlab_format
    parse_reference_parameters

Datatypes
+++++++++

.. autosummary::
   :toctree: generated/data
   :template: namedtuple.rst

    MobilisedTestData
    MobilisedMetadata
    ParsedReferenceData

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
