Left-Right Classification (LRC)
===============================

.. automodule:: mobgap.laterality
    :no-members:
    :no-inherited-members:


Algorithms
++++++++++
.. currentmodule:: mobgap.laterality

.. autosummary::
   :toctree: generated/laterality
   :template: class.rst

    LrcMcCamley
    LrcUllrich

Pipelines
+++++++++
.. automodule:: mobgap.laterality
    :no-members:
    :no-inherited-members:

    pipeline.LrcEmulationPipeline

Evaluation Scores
+++++++++++++++++
These scores are expected to be used in combination with :class:`~mobgap.utils.evaluation.Evaluation` and
:class:`~mobgap.utils.evaluation.EvaluationCV` or directly with :func:`~tpcp.validation.cross_validation` and
:func:`~tpcp.validation.validation`.

.. currentmodule:: mobgap.laterality.evaluation

.. autosummary::
   :toctree: generated/initial_contacts

    lrc_score

.. autosummary::
   :toctree: generated/initial_contacts
   :template: func.rst

    lrc_per_datapoint_score
    lrc_final_agg

Base Classes
++++++++++++
.. automodule:: mobgap.laterality.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.laterality.base

.. autosummary::
   :toctree: generated/laterality
   :template: class.rst

    BaseLRClassifier

Docu-helper
-----------

.. autosummary::
   :toctree: generated/laterality
   :template: func.rst

    base_lrc_docfiller
