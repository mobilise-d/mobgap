Cadence Estimation (CAD)
========================

.. automodule:: mobgap.cadence
    :no-members:
    :no-inherited-members:


Cadence from ICs
++++++++++++++++
.. currentmodule:: mobgap.cadence

.. autosummary::
   :toctree: generated/cadence
   :template: class.rst

    CadFromIc
    CadFromIcDetector

Base Classes
++++++++++++
.. automodule:: mobgap.cadence.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.cadence.base

.. autosummary::
   :toctree: generated/cadence
   :template: class.rst

    BaseCadCalculator

Docu-helper
-----------

.. autosummary::
   :toctree: generated/cadence
   :template: func.rst

    base_cad_docfiller

Pipelines
+++++++++
.. currentmodule:: mobgap.cadence

.. autosummary::
   :toctree: generated/cadence
   :template: class.rst

    pipeline.CadEmulationPipeline

Evaluation
++++++++++
As the structure of the Cadence output is very similar to the output of the full pipeline, we recommend using the
pipeline level evaluation functions to create custom evaluations.

Evaluation Scores
+++++++++++++++++
These scores are expected to be used in combination with :class:`~mobgap.utils.evaluation.Evaluation` and
:class:`~mobgap.utils.evaluation.EvaluationCV` or directly with :func:`~tpcp.validation.cross_validation` and
:func:`~tpcp.validation.validation`.

.. currentmodule:: mobgap.cadence.evaluation

.. autosummary::
   :toctree: generated/cadence

    cad_score

.. autosummary::
   :toctree: generated/cadence
   :template: func.rst

    cad_per_datapoint_score
    cad_final_agg
