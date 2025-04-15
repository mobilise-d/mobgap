"""Dummy algorithm for gsd.

Note, that we moved it here instead of the example file itself is that we need the Dummy algorithm in the Full Pipeline
experiment as well.
"""

from pathlib import Path
from typing import Any, Literal, Optional, Unpack

import pandas as pd
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.utils.conversions import as_samples
from tpcp.caching import hybrid_cache
from typing_extensions import Self


def load_old_gsd_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_csv(result_file_path).astype({"participant_id": str})
    data = data.set_index(data.columns[:-2].to_list())
    return data


class DummyGsdAlgo(BaseGsDetector):
    """A dummy algorithm that responds with the precomputed results of the old pipeline.
    This makes it convenient to compare the results of the old pipeline with the new pipeline, as we can simply use
    the same code to evaluate both.
    However, this also makes things a lot slower compared to just loading all the results, as we need to open the
    file repeatedly and also reload the data from the matlab files, even though the dummy algorithm does not need it.

    Parameters
    ----------
    old_algo_name
        Name of the algorithm for which we want to load the results.
        This determines the name of the file to load.
    base_result_folder
        Base folder where the results are stored.
    min_gs_duration_s
        Minimum duration of a gait sequence in seconds to be considered valid.
        Gait sequences shorter than this duration will be removed from the results.
    """

    def __init__(
        self,
        old_algo_name: str,
        base_result_folder: Path,
        *,
        min_gs_duration_s: float = 0,
    ) -> None:
        self.old_algo_name = old_algo_name
        self.base_result_folder = base_result_folder
        self.min_gs_duration_s = min_gs_duration_s

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        measurement_condition: Optional[
            Literal["free_living", "laboratory"]
        ] = None,
        dp_group: Optional[tuple[str, ...]] = None,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """ "Run" the algorithm."""
        assert measurement_condition is not None, (
            "measurement_condition must be provided"
        )
        assert dp_group is not None, "dp_group must be provided"

        cached_load_old_gsd_results = hybrid_cache(lru_cache_maxsize=1)(
            load_old_gsd_results
        )

        all_results = cached_load_old_gsd_results(
            self.base_result_folder
            / measurement_condition
            / f"{self.old_algo_name}.csv"
        )

        unique_label = dp_group[:-2]
        try:
            gs_list = all_results.loc[unique_label].copy()
        except KeyError:
            gs_list = pd.DataFrame(columns=["start", "end"]).astype(
                {"start": int, "end": int}
            )

        # For some reason some algorithms provide start and end values that are larger than the length of the signal.
        # We clip them here.
        gs_list["end"] = gs_list["end"].clip(upper=len(data))
        # And then we remove the ones where the end is smaller or equal to start.
        gs_list = gs_list[gs_list["start"] < gs_list["end"]]

        # We also remove the ones that are too short.
        gs_list = gs_list[
            gs_list["end"] - gs_list["start"]
            >= as_samples(self.min_gs_duration_s, sampling_rate_hz)
        ]
        self.gs_list_ = gs_list.rename_axis("gs_id").copy()
        return self
