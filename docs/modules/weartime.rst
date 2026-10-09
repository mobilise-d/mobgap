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
    calculate_wtd_classification_summary

Interval utilities
++++++++++++++++++
.. currentmodule:: mobgap.weartime.utils

.. autosummary::
   :toctree: generated/weartime
   :template: func.rst

    clip_intervals_to_waking_hours

Optimization presets
++++++++++++++++++++

The trainable detectors provide dataset-specific settings for
:class:`~mobgap.weartime.optimization.WearTimeOptunaSearch`. For SUSTAIN,
``OptimizationPresets.sustain_weartime`` supplies the search space, wear-time
scorer, combined accuracy ranking and three participant-grouped inner folds.
The optimizer defaults to 20 trials, seed 42 and final refitting. Its default
training transform is identity; dataset selection and subsampling stay with
callers.

.. code-block:: python

    from mobgap.weartime.optimization import WearTimeOptunaSearch
    from mobgap.weartime import WtdMegaritisXGBoost
    from mobgap.weartime.pipeline import WtdEmulationPipeline

    pipeline = WtdEmulationPipeline(
        WtdMegaritisXGBoost(
            **WtdMegaritisXGBoost.PredefinedParameters.untrained_lightweight
        )
    )
    optimizer = WearTimeOptunaSearch(
        pipeline=pipeline,
        **WtdMegaritisXGBoost.OptimizationPresets.sustain_weartime,
    )

Use an explicit untrained ``MegaritisCnnWeartimeModel`` in the pipeline for
``WtdMegaritisCNN.OptimizationPresets.sustain_weartime``. This CNN preset searches
Adam learning rate on a log scale from 0.0001 to 0.01, dropout at 0.2, 0.3 or 0.5,
and training batch size at 256, 512 or 1024. Architecture, windows and epochs
stay configured on the model. These are practical initial search ranges, not
ranges derived from a published tuning study. XGBoost retains the six search
ranges previously configured in the LOSO script.

Override ``cv`` or ``train_dataset_transform`` to customize inner training and
validation. The optimizer does not select recording types. Its shared objective
ranks the mean validation score across inner folds; final refitting uses the
complete dataset passed to ``optimize``.

.. currentmodule:: mobgap.weartime.optimization

.. autosummary::
   :toctree: generated/weartime
   :template: class.rst

    WearTimeOptunaSearch

This draft's earlier ``mobgap.utils.optimization.OptimizableOptunaSearch`` API
has been replaced, with maintainer approval, by the wear-time-specific class
above. Update the import and class name; custom objectives and study factories
are no longer accepted. Configure the scorer, ranking metric and optimization
``direction`` instead. Importing this module requires the optional Optuna dependency.
