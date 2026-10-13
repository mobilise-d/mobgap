"""Optuna search for trainable wear-time pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import optuna
from tpcp import cf
from tpcp.optimize import Optimize
from tpcp.optimize.optuna import CustomOptunaOptimize
from tpcp.validate import cross_validate

from mobgap.weartime.evaluation import wtd_score

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from optuna import Study, Trial
    from optuna.trial import FrozenTrial
    from tpcp.optimize.optuna import StudyParamsDict
    from tpcp.validate import BaseDatasetSplitter, Scorer
    from tpcp.validate._scorer import ScoreType

    from mobgap.data.base import BaseGaitDataset
    from mobgap.weartime.cnn import WtdMegaritisCNNTrainer
    from mobgap.weartime.xgboost import WtdMegaritisXGBoostTrainer

    WearTimeTrainer = WtdMegaritisXGBoostTrainer | WtdMegaritisCNNTrainer


def _seeded_study_params(seed: int) -> StudyParamsDict:
    return {"direction": "maximize", "sampler": optuna.samplers.TPESampler(seed=seed)}


class WearTimeOptunaSearch(CustomOptunaOptimize["WearTimeTrainer", "BaseGaitDataset"]):
    """Search parameters with inner CV, then refit the winner on the provided dataset.

    ``create_search_space`` calls Optuna's ``trial.suggest_*`` methods using pipeline parameter names.
    The shared objective fits each inner training fold with :class:`~tpcp.optimize.Optimize` and averages
    ``score_name`` across its validation folds. ``train_dataset_transform`` changes only these inner
    training sets. The final refit uses the complete dataset provided to ``optimize``.

    Dataset selection belongs to the supplied ``cv`` and ``train_dataset_transform``.

    The default study maximizes the score with seeded TPE sampling. An integer ``cv`` uses ordinary
    splitting; pass a grouped splitter when datapoints from one participant must stay together.
    This module requires the optional Optuna dependency.

    Examples
    --------
    Use the SUSTAIN search preset with a trainer:

    >>> from mobgap.weartime.xgboost import (
    ...     WtdMegaritisXGBoost,
    ...     WtdMegaritisXGBoostTrainer,
    ... )
    >>> pipeline = WtdMegaritisXGBoostTrainer(
    ...     WtdMegaritisXGBoost(
    ...         **WtdMegaritisXGBoost.PredefinedParameters.empty_xgboost
    ...     )
    ... )
    >>> optimizer = WearTimeOptunaSearch(
    ...     pipeline,
    ...     **WtdMegaritisXGBoostTrainer.HyperparameterSearchPresets.sustain_weartime,
    ... )

    The preset supplies human-only, participant-grouped inner validation and a
    seeded training transform that samples human and simulated non-wear days.
    Supply enough simulated days for the configured sample size. These settings
    limit computation; they are not a validated tuning protocol.
    ``train_dataset_transform=None`` keeps complete inner training subsets.
    Named splitter parts allow direct overrides:

    >>> optimizer.set_params(
    ...     cv__parts__human__splitter__base_splitter=5
    ... )  # doctest: +SKIP

    Replace ``cv__parts__human__selector`` to change human-day selection, or
    ``cv__parts__simulated_non_wear__splitter__train`` to change the simulated
    training pool. For CNN, pass :class:`~mobgap.weartime.cnn.WtdMegaritisCNNTrainer` with
    ``WtdMegaritisCNNTrainer.HyperparameterSearchPresets.sustain_weartime``.

    Parameters
    ----------
    pipeline
        Wear-time trainer pipeline. Each trial and CV fold uses a clone.
    create_search_space
        Callable that suggests parameters on the trial, using pipeline parameter names.
    scoring
        Wear-time per-datapoint scorer or Scorer object, defaulting to ``wtd_score``.
    score_name
        Aggregate validation metric to rank. The shared objective averages
        ``test__agg__<score_name>`` across folds. The default ranks ``combined__accuracy``.
        Set ``"score"`` when using a scalar scorer.
    cv
        Inner CV splitter or fold count. An integer uses ordinary splitting.
        Supply a grouped splitter for participant separation.
    train_dataset_transform
        Optional transform of each inner training subset before fitting. ``None`` keeps it unchanged.
        Validation subsets and the final complete-data refit are unaffected.
    get_study_params
        Callable receiving the seed and returning Optuna study arguments. The default maximizes
        the selected metric with seeded TPE. Supply a factory to configure direction, storage
        or other study settings.
    n_trials, random_seed
        Trial budget and sampler seed.
    n_jobs
        Parallel trial workers. The shared objective always runs inner CV sequentially.
        Multiple trial workers require persistent study storage supplied by ``get_study_params``;
        see :class:`~tpcp.optimize.optuna.CustomOptunaOptimize`.
    return_optimized
        Refit the best pipeline on the complete optimization dataset when True.
    timeout, callbacks, gc_after_trial, eval_str_paras, show_progress_bar
        Passed unchanged to :class:`~tpcp.optimize.optuna.CustomOptunaOptimize`; see its documentation
        for stopping, callbacks, cleanup, parameter conversion and progress settings.
    """

    def __init__(
        self,
        pipeline: WearTimeTrainer,
        create_search_space: Callable[[Trial], None],
        *,
        scoring: Scorer[WearTimeTrainer, BaseGaitDataset]
        | Callable[[WearTimeTrainer, BaseGaitDataset], ScoreType] = cf(wtd_score),
        score_name: str = "combined__accuracy",
        cv: BaseDatasetSplitter | int = 3,
        train_dataset_transform: Callable[[BaseGaitDataset], BaseGaitDataset] | None = None,
        get_study_params: Callable[[int], StudyParamsDict] = _seeded_study_params,
        n_trials: int = 50,
        random_seed: int = 42,
        timeout: float | None = None,
        callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
        gc_after_trial: bool = False,
        n_jobs: int = 1,
        eval_str_paras: Sequence[str] = (),
        show_progress_bar: bool = False,
        return_optimized: bool = True,
    ) -> None:
        self.create_search_space = create_search_space
        self.scoring = scoring
        self.score_name = score_name
        self.cv = cv
        self.train_dataset_transform = train_dataset_transform
        super().__init__(
            pipeline,
            get_study_params,
            n_trials=n_trials,
            random_seed=random_seed,
            timeout=timeout,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            n_jobs=n_jobs,
            eval_str_paras=eval_str_paras,
            show_progress_bar=show_progress_bar,
            return_optimized=return_optimized,
        )

    def create_objective(self) -> Callable[[Trial, WearTimeTrainer, BaseGaitDataset], float]:
        """Rank candidates using the configured inner CV objective."""

        def objective(trial: Trial, candidate: WearTimeTrainer, dataset: BaseGaitDataset) -> float:
            self.create_search_space(trial)
            candidate.set_params(**self.sanitize_params(trial.params))
            scores = cross_validate(
                Optimize(candidate, train_dataset_transform=self.train_dataset_transform),
                dataset,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=1,
                return_train_score=False,
                progress_bar=False,
            )
            return float(np.mean(scores[f"test__agg__{self.score_name}"]))

        return objective


__all__ = ["WearTimeOptunaSearch"]
