"""Inject a trial objective into TPCP's optimizable Optuna search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tpcp.optimize.optuna import CustomOptunaOptimize

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from optuna import Study, Trial
    from optuna.trial import FrozenTrial
    from tpcp import Dataset, OptimizablePipeline


class OptimizableOptunaSearch(CustomOptunaOptimize):
    """Run an injected objective and let TPCP refit the best pipeline."""

    def __init__(
        self,
        pipeline: OptimizablePipeline,
        get_study_params: Callable[[int], dict[str, Any]],
        objective: Callable[[Trial, OptimizablePipeline, Dataset], float],
        *,
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
        """Use the objective supplied by the caller."""
        return self.objective
