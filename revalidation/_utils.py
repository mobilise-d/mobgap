"""Shared helpers for revalidation scripts."""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from tpcp.misc import iter_with_warning_error_context
from tpcp.parallel import delayed


def create_evaluation_tasks(
    run_evaluation: Callable[[str, Any, Any], tuple[str, Any]],
    pipelines: Mapping[str, Any],
    dataset: Any,
    *,
    condition: str,
) -> Iterator[Any]:
    """Create context-aware delayed evaluation tasks."""
    for make_context, (name, pipeline) in iter_with_warning_error_context(
        pipelines.items()
    ):
        with make_context(
            "revalidation", {"pipeline": name, "condition": condition}
        ):
            task = delayed(run_evaluation)(name, pipeline, dataset)
        yield task
