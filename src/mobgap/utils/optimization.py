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
