"""Reusable Optuna search for optimizable TPCP pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
from tpcp.optimize import Optimize
from tpcp.optimize.optuna import CustomOptunaOptimize
from tpcp.validate import cross_validate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from optuna import Study, Trial
    from optuna.trial import FrozenTrial
    from tpcp import Dataset, OptimizablePipeline
    from tpcp.validate import BaseDatasetSplitter, Scorer


def _seeded_study(seed: int) -> dict[str, Any]:
    return {"direction": "maximize", "sampler": optuna.samplers.TPESampler(seed=seed)}


class OptimizableOptunaSearch(CustomOptunaOptimize):
    """Search parameters with inner CV, then refit the winner on the provided dataset.

    ``create_search_space`` calls Optuna's ``trial.suggest_*`` methods using pipeline parameter names.
    The shared objective fits each inner training fold with :class:`~tpcp.optimize.Optimize` and averages
    ``score_name`` across its validation folds. ``train_dataset_transform`` changes only these inner
    training sets. The final refit uses the complete dataset provided to ``optimize``.

    Supply ``objective`` to replace the shared CV objective. It receives the trial, cloned pipeline
    and optimization dataset; its suggested parameter names must also be pipeline parameter names.
    Dataset selection belongs to the supplied ``cv`` and ``train_dataset_transform``.

    The default study maximizes the score with seeded TPE sampling. ``cv=3`` uses ordinary three-fold
    splitting; pass a grouped splitter when datapoints from one participant must stay together.
    This module requires the optional Optuna dependency.

    Parameters
    ----------
    pipeline
        Trainable TPCP pipeline. Each trial and CV fold uses a clone.
    create_search_space
        Callable that suggests parameters on the trial, using pipeline parameter names.
        Required for the shared CV objective; ignored when ``objective`` is supplied.
    scoring
        Per-datapoint scorer or Scorer object. Required for the shared CV objective.
        Ignored when ``objective`` is supplied, which may leave it as ``None``.
    score_name
        Aggregate validation metric to rank. The shared objective averages
        ``test__agg__<score_name>`` across folds. Scalar scorers use the default ``"score"``.
        Ignored when ``objective`` is supplied.
    cv
        Inner CV splitter or fold count. The default integer uses ordinary three-fold splitting.
        Supply a grouped splitter for participant separation. Ignored by an injected objective.
    train_dataset_transform
        Optional transform of each inner training subset before fitting. ``None`` keeps it unchanged.
        Validation subsets and the final complete-data refit are unaffected. An injected objective
        owns its own training procedure and does not apply this transform automatically.
    objective
        Optional replacement objective receiving ``(trial, cloned_pipeline, dataset)``.
        It owns parameter suggestions, training and scoring. Final refitting still uses its winning
        pipeline parameters and the complete optimization dataset.
    get_study_params
        Callable receiving the seed and returning Optuna study arguments. The default maximizes
        the score with a seeded TPE sampler.
    n_trials, random_seed
        Trial budget and sampler seed, defaulting to 20 and 42.
    n_jobs
        Parallel trial workers. The shared objective always runs inner CV sequentially.
        Multiple trial workers require persistent study storage, as described in
        :class:`~tpcp.optimize.optuna.CustomOptunaOptimize`.
    return_optimized
        Refit the best pipeline on the complete optimization dataset when True.
    timeout, callbacks, gc_after_trial, eval_str_paras, show_progress_bar
        Passed unchanged to :class:`~tpcp.optimize.optuna.CustomOptunaOptimize`; see its documentation
        for stopping, callbacks, cleanup, parameter conversion and progress settings.
    """

    def __init__(
        self,
        pipeline: OptimizablePipeline,
        create_search_space: Callable[[Trial], None] | None = None,
        *,
        scoring: Scorer | Callable | None = None,
        score_name: str = "score",
        cv: BaseDatasetSplitter | int = 3,
        train_dataset_transform: Callable[[Dataset], Dataset] | None = None,
        objective: Callable[[Trial, OptimizablePipeline, Dataset], float] | None = None,
        get_study_params: Callable[[int], dict[str, Any]] = _seeded_study,
        n_trials: int = 20,
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
        self.objective = objective
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

    def create_objective(self) -> Callable[[Trial, OptimizablePipeline, Dataset], float]:
        """Use the injected objective or the configured inner CV objective."""
        if self.objective is not None:
            return self.objective

        def objective(trial: Trial, candidate: OptimizablePipeline, dataset: Dataset) -> float:
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


__all__ = ["OptimizableOptunaSearch"]
