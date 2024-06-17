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
    GsdIonescu
    GsdAdaptiveIonescu

Pipelines
+++++++++
.. currentmodule:: mobgap.gsd

.. autosummary::
   :toctree: generated/gsd
   :template: class.rst

    pipeline.GsdEmulationPipeline

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
    categorize_intervals_per_sample
    categorize_intervals
    plot_categorized_intervals
    get_matching_intervals
