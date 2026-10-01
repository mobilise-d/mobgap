Wear-time detection
===================

.. automodule:: mobgap.weartime
    :no-members:
    :no-inherited-members:

Algorithms
++++++++++
.. currentmodule:: mobgap.weartime

.. autosummary::
   :toctree: generated/weartime
   :template: class.rst

    WtdMegaritisSignal

Base classes
++++++++++++
.. currentmodule:: mobgap.weartime.base

.. autosummary::
   :toctree: generated/weartime
   :template: class.rst

    BaseWeartimeDetector

Pipelines
+++++++++
.. currentmodule:: mobgap.weartime.pipeline

.. autosummary::
   :toctree: generated/weartime
   :template: class.rst

    WtdEmulationPipeline

Evaluation
++++++++++
.. currentmodule:: mobgap.weartime.evaluation

.. autosummary::
   :toctree: generated/weartime

    wtd_score

.. autosummary::
   :toctree: generated/weartime
   :template: func.rst

    wtd_per_datapoint_score
    wtd_final_agg

Interval utilities
++++++++++++++++++
.. currentmodule:: mobgap.weartime.utils

.. autosummary::
   :toctree: generated/weartime
   :template: func.rst

    clip_intervals_to_waking_hours
