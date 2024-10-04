from pathlib import Path
from typing import Any, Self, Unpack

import pandas as pd
from joblib import Memory, Parallel
from tpcp.caching import hybrid_cache
from tpcp.parallel import delayed

from mobgap import PACKAGE_ROOT
from mobgap.data import TVSLabDataset
from mobgap.gait_sequences import GsdIluz, GsdIonescu, GsdAdaptiveIonescu
from mobgap.gait_sequences._evaluation_challenge import (
    GsdEvaluation,
    gsd_evaluation_scorer,
)
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline


def load_old_gsd_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_json(result_file_path, orient="index")
    data.index = pd.MultiIndex.from_tuples(data.index.map(eval))
    return data


class DummyGsdAlgo(BaseGsDetector):
    def __init__(self, old_algo_name: str):
        self.old_algo_name = old_algo_name

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        raise NotImplementedError


class DummyGsdEvaluationPipeline(GsdEmulationPipeline):
    def __init__(
        self,
        algo: BaseGsDetector,
        *,
        convert_to_body_frame: bool = True,
        base_result_folder: Path,
    ) -> None:
        self.base_result_folder = base_result_folder
        super().__init__(algo, convert_to_body_frame=convert_to_body_frame)

    def run(self, datapoint):
        if not isinstance(self.algo, DummyGsdAlgo):
            raise ValueError(f"Expected DummyGsdAlgo, got {self.algo}")

        # Instead of running anything, we just load the results from the original GSD output.
        self.algo_ = self.algo.clone()

        cached_load_old_gsd_results = hybrid_cache(lru_cache_maxsize=1)(load_old_gsd_results)

        all_results = cached_load_old_gsd_results(
            self.base_result_folder
            / datapoint.recording_metadata["measurement_condition"]
            / f"{self.algo.old_algo_name}.zip"
        )
        try:
            gs_list = all_results.loc[datapoint.group_label]
        except KeyError:
            gs_list = pd.DataFrame(columns=["start", "end"]).astype(
                {"start": int, "end": int}
            )

        # For some reason some algorithms provide start and end values that are larger than the length of the signal.
        # We clip them here.
        gs_list["end"] = gs_list["end"].clip(upper=len(datapoint.data_ss))
        # And then we remove the ones were the end is smaller or equal to start.
        gs_list = gs_list[gs_list["start"] < gs_list["end"]]

        self.algo_.gs_list_ = gs_list.rename_axis("gs_id").copy()

        return self


tvs_labdataset = TVSLabDataset(
    "/home/arne/Documents/repos/private/mobilised_tvs_data/tvs_dataset/",
    reference_system="INDIP",
    memory=Memory(PACKAGE_ROOT.parent / ".cache"),
    missing_reference_error_type="skip",
)


dummy_pipe = DummyGsdEvaluationPipeline(
    DummyGsdAlgo("Gaitpy"),
    base_result_folder=Path(
        "/home/arne/Documents/repos/private/mobgap_validation/data/gsd/"
    ),
)




pipelines = {}
# for name in ['EPFL_V1-improved_th', 'EPFL_V1-original', 'EPFL_V2-original', 'Gaitpy', 'Hickey-original', 'Rai', 'TA_Iluz-original', 'TA_Wavelets_v2']:
for name in ['TA_Iluz-original', 'TA_Wavelets_v2']:
    pipelines[name] = DummyGsdEvaluationPipeline(
        DummyGsdAlgo(name),
        base_result_folder=Path(
            "/home/arne/Documents/repos/private/mobgap_validation/data/gsd/"
        ),
    )

# for algo in (GsdIluz(), GsdIonescu(), GsdAdaptiveIonescu()):
#     pipelines[algo.__class__.__name__] = GsdEmulationPipeline(algo)

def run_evaluation(name, pipeline, dataset):
    print(name)
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