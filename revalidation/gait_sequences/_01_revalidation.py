import warnings
from pathlib import Path
from typing import Any, Optional, Self, Unpack

import pandas as pd
from joblib import Memory, Parallel
from mobgap import PACKAGE_ROOT
from mobgap.data import TVSLabDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu, GsdIluz, GsdIonescu
from mobgap.gait_sequences.base import BaseGsDetector
from mobgap.gait_sequences.evaluation import (
    gsd_evaluation_scorer,
)
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline
from mobgap.utils.misc import get_env_var
from tpcp.caching import hybrid_cache
from tpcp.parallel import delayed


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

    """

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
        """ "Run" the algorithm."""
        assert (
            measurement_condition is not None
        ), "measurement_condition must be provided"
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

        self.gs_list_ = gs_list.rename_axis("gs_id").copy()
        return self


tvs_labdataset = TVSLabDataset(
    get_env_var("MOBGAP_TVS_DATASET_PATH"),
    reference_system="INDIP",
    memory=Memory(PACKAGE_ROOT.parent / ".cache"),
    missing_reference_error_type="skip",
).get_subset(cohort=["PFF", "HA", "CHF"])

old_data_path = Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "data/gsd"

pipelines = {}
for name in [
    "EPFL_V1-improved_th",
    "EPFL_V1-original",
    "EPFL_V2-original",
    "Gaitpy",
    "Hickey-original",
    "Rai",
    "TA_Iluz-original",
    "TA_Wavelets_v2",
]:
    pipelines[name] = GsdEmulationPipeline(
        DummyGsdAlgo(name, base_result_folder=old_data_path)
    )

for algo in (GsdIluz(), GsdIonescu(), GsdAdaptiveIonescu()):
    pipelines[algo.__class__.__name__] = GsdEmulationPipeline(algo)


pipelines["GSDIluz"] = GsdEmulationPipeline(
    GsdIluz(use_original_peak_detection=False)
)
pipelines["GSDIluz_orig_peak"] = GsdEmulationPipeline(
    GsdIluz(use_original_peak_detection=True)
)

# %%
from mobgap.gait_sequences.evaluation import GsdEvaluation


def run_evaluation(name, pipeline, dataset):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Zero division", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore", message="multiple ICs", category=UserWarning
        )
        eval_pipe = GsdEvaluation(
            dataset,
            scoring=gsd_evaluation_scorer,
        ).run(pipeline)
    return name, eval_pipe.results_


with Parallel(n_jobs=3) as parallel:
    results = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, tvs_labdataset)
            for name, pipeline in pipelines.items()
        )
    )


# %%
def extract_single_performance(result):
    result = pd.DataFrame(result)
    single_cols = result.filter(like="single_").columns
    result = result[["data_labels", *single_cols]]
    result = result.explode(result.columns.to_list()).set_index("data_labels")
    result.columns = [c.split("__")[-1] for c in result.columns]
    result.index = pd.MultiIndex.from_tuples(result.index)
    return result


# %%
import matplotlib.pyplot as plt
import seaborn as sns

results_df = (
    pd.concat({k: extract_single_performance(v) for k, v in results.items()})
    .reset_index()
    .rename(columns={"level_0": "algo_name"})
)

sns.boxplot(data=results_df, x="cohort", y="accuracy", hue="algo_name")

plt.show()
