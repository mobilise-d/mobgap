Gait Sequence Detection (GSD)
=============================

.. automodule:: mobgap.gsd
    :no-members:
    :no-inherited-members:


Algorithms
++++++++++
.. currentmodule:: mobgap.gsd

.. autosummary::
   :toctree: generated/gsd
   :template: class.rst

    GsdIluz
    GsdParaschivIonescu

Pipelines
+++++++++
.. currentmodule:: mobgap.gsd

.. autosummary::
   :toctree: generated/gsd
   :template: class.rst

    evaluation.GsdEvaluationPipeline

Base Classes
++++++++++++
.. automodule:: mobgap.gsd.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.gsd.base

.. autosummary::
   :toctree: generated/gsd
   :template: class.rst

    BaseGsDetector

Docu-helper
-----------

.. autosummary::
   :toctree: generated/gsd
   :template: func.rst

    base_gsd_docfiller


Evaluation
++++++++++
.. currentmodule:: mobgap.gsd.evaluation

.. autosummary::
   :toctree: generated/gsd
   :template: func.rst

    calculate_matched_gsd_performance_metrics
    calculate_unmatched_gsd_performance_metrics
    categorize_intervals
    find_matches_with_min_overlap
    plot_categorized_intervals

