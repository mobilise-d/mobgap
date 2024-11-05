from pathlib import Path
from typing import Any, Self, Unpack, Optional

import pandas as pd
from joblib import Memory, Parallel
import warnings

from mobgap import PACKAGE_ROOT
from mobgap.data import TVSLabDataset
from mobgap.gait_sequences.evaluation import (
    GsdEvaluation,
    gsd_evaluation_scorer,
)
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline
from tpcp.caching import hybrid_cache
from tpcp.parallel import delayed


def load_old_gsd_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_json(result_file_path, orient="records", lines=True)
    data.index = pd.MultiIndex.from_tuples(data["id"])
    return data.drop(columns="id")


class DummyGsdAlgo(BaseGsDetector):
    def __init__(self, old_algo_name: str, base_result_folder: Path) -> None:
        self.old_algo_name = old_algo_name
        self.base_result_folder = base_result_folder

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
            measurement_condition: Optional[str] = None,
        dp_group: Optional[tuple[str, ...]] = None,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        assert measurement_condition is not None, "measurement_condition must be provided"
        assert dp_group is not None, "dp_group must be provided"

        cached_load_old_gsd_results = hybrid_cache(lru_cache_maxsize=1)(
            load_old_gsd_results
        )

        all_results = cached_load_old_gsd_results(
            self.base_result_folder
            / measurement_condition
            / f"{self.old_algo_name}.json"
        )
        try:
            gs_list = all_results.loc[dp_group]
        except KeyError:
            gs_list = pd.DataFrame(columns=["start", "end"]).astype(
                {"start": int, "end": int}
            )

        # For some reason some algorithms provide start and end values that are larger than the length of the signal.
        # We clip them here.
        gs_list["end"] = gs_list["end"].clip(upper=len(data))
        # And then we remove the ones where the end is smaller or equal to start.
        gs_list = gs_list[gs_list["start"] < gs_list["end"]]

        self.gs_list_ = gs_list.rename_axis("gs_id").copy()
        return self


tvs_labdataset = TVSLabDataset(
    "/home/arne/Documents/repos/private/mobilised_tvs_data/tvs_dataset/",
    reference_system="INDIP",
    memory=Memory(PACKAGE_ROOT.parent / ".cache"),
    missing_reference_error_type="skip",
)


pipelines = {}
# for name in ['EPFL_V1-improved_th', 'EPFL_V1-original', 'EPFL_V2-original', 'Gaitpy', 'Hickey-original', 'Rai', 'TA_Iluz-original', 'TA_Wavelets_v2']:
for name in ["TA_Iluz-original", "TA_Wavelets_v2"]:
    pipelines[name] = GsdEmulationPipeline(
        DummyGsdAlgo(name,
        base_result_folder=Path(
            "/home/arne/Documents/repos/private/mobgap_validation/data/gsd/"
        ),
                     )
    )

# for algo in (GsdIluz(), GsdIonescu(), GsdAdaptiveIonescu()):
#     pipelines[algo.__class__.__name__] = GsdEmulationPipeline(algo)


def run_evaluation(name, pipeline, dataset):
    print(name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Zero division", category=UserWarning)
        warnings.filterwarnings("ignore", message="multiple ICs", category=UserWarning)
        eval_pipe = GsdEvaluation(
            dataset,
            scoring=gsd_evaluation_scorer,
        ).run(pipeline)
    return eval_pipe.results_


with Parallel(n_jobs=3) as parallel:
    results = parallel(
        delayed(run_evaluation)(name, pipeline, tvs_labdataset)
        for name, pipeline in pipelines.items()
    )