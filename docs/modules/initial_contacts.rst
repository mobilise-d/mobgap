Initial Contact Detection (ICD)
===============================

.. automodule:: mobgap.initial_contacts
    :no-members:
    :no-inherited-members:


Algorithms
++++++++++
.. currentmodule:: mobgap.initial_contacts

.. autosummary::
   :toctree: generated/initial_contacts
   :template: class.rst

    IcdShinImproved
    IcdIonescu
    IcdHKLeeImproved

Pipelines
+++++++++
.. currentmodule:: mobgap.initial_contacts

.. autosummary::
   :toctree: generated/initial_contacts
   :template: class.rst

    pipeline.IcdEmulationPipeline

Utils
+++++
.. currentmodule:: mobgap.initial_contacts

.. autosummary::
   :toctree: generated/initial_contacts
   :template: function.rst

    refine_gs

Evaluation
++++++++++
.. currentmodule:: mobgap.initial_contacts.evaluation

.. autosummary::
   :toctree: generated/initial_contacts
   :template: func.rst

    calculate_matched_icd_performance_metrics
    calculate_true_positive_icd_error
    categorize_ic_list
    get_matching_ics

Evaluation Scores
+++++++++++++++++
These scores are expected to be used in combination with :class:`~mobgap.utils.evaluation.Evaluation` and
:class:`~mobgap.utils.evaluation.EvaluationCV` or directly with :func:`~tpcp.validation.cross_validation` and
:func:`~tpcp.validation.validation`.

.. currentmodule:: mobgap.initial_contacts.evaluation

.. autosummary::
   :toctree: generated/initial_contacts

    icd_score

.. autosummary::
   :toctree: generated/initial_contacts
   :template: func.rst

    icd_per_datapoint_score
    icd_final_agg

Base Classes
++++++++++++
.. automodule:: mobgap.initial_contacts.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.initial_contacts.base

.. autosummary::
   :toctree: generated/initial_contacts
   :template: class.rst

    BaseIcDetector

Docu-helper
-----------

.. autosummary::
   :toctree: generated/initial_contacts
   :template: func.rst

    base_icd_docfiller

