Pipelines
=========

.. automodule:: mobgap.pipeline
    :no-members:
    :no-inherited-members:

Full Pipelines
--------------

.. currentmodule:: mobgap.pipeline

.. autosummary::
   :toctree: generated/pipeline
   :template: class.rst

    BaseMobilisedPipeline
    MobilisedPipelineHealthy
    MobilisedPipelineImpaired


Evaluation
----------

WB-Matching
+++++++++++
.. currentmodule:: mobgap.pipeline.evaluation

.. autosummary::
   :toctree: generated/pipeline
   :template: func.rst

    categorize_intervals
    categorize_intervals_per_sample
    get_matching_intervals


Per-Row-Error Funcs
+++++++++++++++++++
.. currentmodule:: mobgap.pipeline.evaluation

.. autosummary::
   :toctree: generated/pipeline.evaluation
   :template: class.rst

    ErrorTransformFuncs

.. autosummary::
   :toctree: generated/pipeline
   :template: func.rst

    get_default_error_transformations
    error
    rel_error
    abs_error
    abs_rel_error

Custom Error Aggregations
+++++++++++++++++++++++++
.. currentmodule:: mobgap.pipeline.evaluation

.. autosummary::
   :toctree: generated/pipeline.evaluation
   :template: class.rst

    CustomErrorAggregations

.. autosummary::
   :toctree: generated/pipeline
   :template: func.rst

    get_default_error_aggregations
    icc
    loa
    quantiles

Helper
------

Gait Sequence Iteration
+++++++++++++++++++++++
.. currentmodule:: mobgap.pipeline

.. autosummary::
   :toctree: generated/pipeline
   :template: class.rst

    GsIterator

.. autosummary::
   :toctree: generated/pipeline
   :template: func.rst

    iter_gs

Aggregation Functions
~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: mobgap.pipeline

.. autosummary::
   :toctree: generated/pipeline
   :template: func.rst

    create_aggregate_df

Datatypes
~~~~~~~~~
.. currentmodule:: mobgap.pipeline

.. autosummary::
   :toctree: generated/pipeline
   :template: class.rst

    FullPipelinePerGsResult

