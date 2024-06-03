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
    categorize_intervals
    categorize_matches_with_min_overlap
    plot_categorized_intervals
    combine_det_with_ref_without_matching
    get_matching_gs
    error
    rel_error
    abs_error
    abs_rel_error
    quantiles
    loa
    icc
    get_default_error_metrics
    get_default_aggregations
    apply_transformations
    apply_aggregations

