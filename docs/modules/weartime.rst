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
    calculate_wtd_simulated_non_wear_summary

Human and simulated non-wear summaries are separate. Simulated non-wear pools
counts per original recording in each fold and averages fold-model metrics for
each recording before equal recording means and 95% t intervals. Repeated
predictions do not add independent CI observations. False-wear minutes are
per evaluated day, without normalization to 24 hours.

Interval utilities
++++++++++++++++++
.. currentmodule:: mobgap.weartime.utils

.. autosummary::
   :toctree: generated/weartime
   :template: func.rst

    clip_intervals_to_waking_hours

Training accepts ``(data, reference_weartime)`` recording pairs. References must
include an unordered categorical ``label`` column with categories
``["wear", "uncertain"]``, including empty and all-wear frames. ``uncertain``
means wear versus non-wear could not be determined. The unlabeled complement
is known non-wear. Windows touching uncertain samples are
excluded from CNN and XGBoost training; intervals use half-open boundaries.
Known windows remain on the original recording grid, including on either side
of an uncertain gap. No signal segments are concatenated.

Optimization presets
++++++++++++++++++++

The shared wear-time revalidation scripts compare signal, CNN and XGBoost
using one outer participant LOSO protocol and a fixed half of complete
simulated non-wear recordings for evaluation. Both trainable detectors use
their SUSTAIN search presets. The separate ``scripts/weartime_ml_training_evaluation/train.py``
entry point tunes on inner CV and refits final classifiers on the complete
provided dataset, without a second outer evaluation protocol.

The trainable detectors provide dataset-specific settings for
:class:`~mobgap.weartime.optimization.WearTimeOptunaSearch`. For SUSTAIN,
``OptimizationPresets.sustain_weartime`` supplies the search space, wear-time
scorer, combined accuracy ranking and three participant-grouped inner folds.
Inner validation includes only human-movement days. Each inner training fold
uses 40% of its human days and five seeded simulated non-wear days from the
provided training pool, sampled independently with seed 42. Supply at least five
simulated non-wear training days; the preset does not select or alter the outer train/test split.
These sampling choices are computational-budget defaults selected for the
training experiments in this repository: 40% reduces human-day preprocessing
and fitting costs, while five simulated non-wear days provide a limited
non-wear training pool. They are not an author-reported or scientifically validated tuning
protocol. Changing them changes the training distribution used to rank
candidates; configure the training transform for the intended experiment.
The optimizer defaults to 20 trials, seed 42 and final refitting. Callers can
override the inner splitter and training transform; ``train_dataset_transform=None``
keeps complete inner training subsets.

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

The preset splitter has named ``human`` and ``simulated_non_wear`` parts.
Override their nested parameters directly, for example:

.. code-block:: python

    optimizer.set_params(
        cv__parts__human__splitter__base_splitter=5,
    )

Use ``cv__parts__human__selector`` to replace the human-day selector, or
``cv__parts__simulated_non_wear__splitter__train`` to select a different simulated
non-wear training pool. ``cv__parts__human`` replaces the complete human child.

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
above. Update the import and class name; custom objectives are no longer
accepted. Configure the scorer and ranking metric for the shared CV objective.
Use the normal TPCP ``get_study_params`` factory parameter to configure study
direction, sampling or storage. Importing this module requires the optional
Optuna dependency.
