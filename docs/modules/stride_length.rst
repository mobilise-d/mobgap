Stride Length (SL)
==================

.. automodule:: mobgap.stride_length
    :no-members:
    :no-inherited-members:


Algorithms
++++++++++
.. currentmodule:: mobgap.stride_length

.. autosummary::
   :toctree: generated/stride_length
   :template: class.rst

    SlZijlstra

Pipelines
+++++++++
.. currentmodule:: mobgap.stride_length

.. autosummary::
   :toctree: generated/stride_length
   :template: class.rst

    pipeline.SlEmulationPipeline

Base Classes
++++++++++++
.. automodule:: mobgap.stride_length.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.stride_length.base

.. autosummary::
   :toctree: generated/stride_length
   :template: class.rst

    BaseSlCalculator

Docu-helper
-----------

.. autosummary::
   :toctree: generated/stride_length
   :template: func.rst

    base_sl_docfiller

Evaluation
++++++++++
As the structure of the Stride length output is very similar to the output of the full pipeline, we recommend using
the pipeline level evaluation functions to create custom evaluations.

Evaluation Scores
+++++++++++++++++
These scores are expected to be used in combination with :class:`~mobgap.utils.evaluation.Evaluation` and
:class:`~mobgap.utils.evaluation.EvaluationCV` or directly with :func:`~tpcp.validation.cross_validation` and
:func:`~tpcp.validation.validation`.

.. currentmodule:: mobgap.stride_length.evaluation

.. autosummary::
   :toctree: generated/stride_length

    sl_score

.. autosummary::
   :toctree: generated/stride_length
   :template: func.rst

    sl_per_datapoint_score
    sl_final_agg
