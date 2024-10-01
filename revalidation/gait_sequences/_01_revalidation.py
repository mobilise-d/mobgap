from functools import lru_cache
from pathlib import Path
from typing import Unpack, Any, Self

import pandas as pd
from joblib import Memory

from mobgap import PACKAGE_ROOT
from mobgap.data import TVSLabDataset
from mobgap.gait_sequences._evaluation_challenge import gsd_evaluation_scorer, GsdEvaluation
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline


# @lru_cache(maxsize=1)
def load_old_gsd_results(result_file_path: Path) -> pd.DataFrame:
    assert result_file_path.exists(), result_file_path
    data = pd.read_json(result_file_path, orient="index")
    data.index = pd.MultiIndex.from_tuples(data.index.map(eval))
    return data


class DummyGsdAlgo(BaseGsDetector):

    def __init__(self, old_algo_name: str):
        self.old_algo_name = old_algo_name

    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        raise NotImplementedError


class DummyGsdEvaluationPipeline(GsdEmulationPipeline):

    def __init__(self, algo: BaseGsDetector, *, convert_to_body_frame: bool = True, base_result_folder: Path) -> None:
        self.base_result_folder = base_result_folder
        super().__init__(algo, convert_to_body_frame=convert_to_body_frame)

    def run(self, datapoint):
        if not isinstance(self.algo, DummyGsdAlgo):
            raise ValueError(f"Expected DummyGsdAlgo, got {self.algo}")

        # Instead of running anything, we just load the results from the original GSD output.
        self.algo_ = self.algo.clone()
        all_results = load_old_gsd_results(self.base_result_folder / datapoint.recording_metadata["measurement_condition"] / f"{self.algo.old_algo_name}.zip")
        try:
            gs_list = all_results.loc[datapoint.group_label]
        except KeyError:
            gs_list = pd.DataFrame(columns=["start", "end"]).astype({"start": int, "end": int})
        self.algo_.gs_list_ = gs_list.rename_axis("gs_id")

        return self


tvs_labdataset = TVSLabDataset("/home/arne/Documents/repos/private/mobilised_tvs_data/tvs_dataset/", reference_system="INDIP",
                               memory=Memory(PACKAGE_ROOT.parent / ".cache"),

                               missing_reference_error_type="skip",)


dummy_pipe = DummyGsdEvaluationPipeline(DummyGsdAlgo("Gaitpy"), base_result_folder=Path("/home/arne/Documents/repos/private/mobgap_validation/data/gsd/"))

if __name__ == "__main__":
    eval_pipe = GsdEvaluation(
        tvs_labdataset,
        scoring=gsd_evaluation_scorer,
        validate_paras={"n_jobs": 3},
    ).run(dummy_pipe)
    results = eval_pipe.results_