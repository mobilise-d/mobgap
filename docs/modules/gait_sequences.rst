Gait Sequence Detection (GSD)
=============================

.. automodule:: mobgap.gait_sequences
    :no-members:
    :no-inherited-members:


Algorithms
++++++++++
.. currentmodule:: mobgap.gait_sequences

.. autosummary::
   :toctree: generated/gait_sequences
   :template: class.rst

    GsdIluz
    GsdIonescu
    GsdAdaptiveIonescu

Pipelines
+++++++++
.. currentmodule:: mobgap.gait_sequences

.. autosummary::
   :toctree: generated/gait_sequences
   :template: class.rst

    pipeline.GsdEmulationPipeline

Base Classes
++++++++++++
.. automodule:: mobgap.gait_sequences.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.gait_sequences.base

.. autosummary::
   :toctree: generated/gait_sequences
   :template: class.rst

    BaseGsDetector

Docu-helper
-----------

.. autosummary::
   :toctree: generated/gait_sequences
   :template: func.rst

    base_gsd_docfiller


Evaluation
++++++++++
.. currentmodule:: mobgap.gait_sequences.evaluation

.. autosummary::
   :toctree: generated/gait_sequences
   :template: func.rst

    calculate_matched_gsd_performance_metrics
    calculate_unmatched_gsd_performance_metrics
    categorize_intervals_per_sample
    categorize_intervals
    plot_categorized_intervals
    get_matching_intervals

Evaluation Scores
+++++++++++++++++
These scores are expected to be used in combination with :class:`~mobgap.utils.evaluation.Evaluation` and
:class:`~mobgap.utils.evaluation.EvaluationCV` or directly with :func:`~tpcp.validation.cross_validation` and
:func:`~tpcp.validation.validation`.

.. currentmodule:: mobgap.gait_sequences.evaluation

.. autosummary::
   :toctree: generated/stride_length

    gsd_score

.. autosummary::
   :toctree: generated/gait_sequences
   :template: func.rst

    gsd_per_datapoint_score
    gsd_final_agg