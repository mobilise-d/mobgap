Data loading and Datasets
=========================

.. automodule:: mobgap.data
    :no-members:
    :no-inherited-members:

General Gait Data
-----------------

Base Classes
++++++++++++
.. currentmodule:: mobgap.data.base

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    BaseGaitDataset
    BaseGaitDatasetWithReference

Generic Loader Classes
++++++++++++++++++++++
.. currentmodule:: mobgap.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    GaitDatasetFromData


Mobilise-D Matlab format
------------------------

Base Classes
++++++++++++
.. currentmodule:: mobgap.data

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


Technical Validation Study (TVS) Data Loader
--------------------------------------------

.. currentmodule:: mobgap.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    TVSLabDataset
    TVSFreeLivingDataset

MS Project Dataset
------------------

.. currentmodule:: mobgap.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    MsProjectDataset


Base Classes
++++++++++++
.. autosummary::
   :toctree: generated/data
   :template: class.rst

    BaseTVSDataset


Example Data
------------

Dataset Classes
+++++++++++++++
.. currentmodule:: mobgap.data

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


Mobilise-D v1.0 Pipeline Result Loaders
---------------------------------------
.. currentmodule:: mobgap.data

.. autosummary::
   :toctree: generated/data
   :template: class.rst

    MobilisedCvsDmoDataset


Datatypes
---------
.. currentmodule:: mobgap.data.base

.. autosummary::
   :toctree: generated/data
   :template: namedtuple.rst

    ReferenceData

.. currentmodule:: mobgap.data

.. autosummary::
   :toctree: generated/data
   :template: namedtuple.rst

    MobilisedTestData

.. autosummary::
   :toctree: generated/data
   :template: typed_dict.rst

    base.ParticipantMetadata
    MobilisedParticipantMetadata
    base.RecordingMetadata
    MobilisedMetadata


Docfiller
---------
.. currentmodule:: mobgap.data.base

.. autosummary::
   :toctree: generated/data
   :template: function.rst

    base_gait_dataset_docfiller

.. currentmodule:: mobgap.data

.. autosummary::
   :toctree: generated/data
   :template: function.rst

    matlab_dataset_docfiller
