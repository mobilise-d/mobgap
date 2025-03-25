from mobgap.laterality import LrcMcCamley, LrcUllrich
from mobgap.laterality.pipeline import LrcEmulationPipeline
from mobgap.utils.misc import get_env_var
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from tpcp.optimize import DummyOptimize, Optimize

pipelines = {
    "McCamley": DummyOptimize(LrcEmulationPipeline(LrcMcCamley())),
    "UllrichOld": DummyOptimize(
        LrcEmulationPipeline(
            LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all_old)
        )
    ),
    "UllrichNew": DummyOptimize(
        LrcEmulationPipeline(
            LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all)
        )
    ),
    "UllrichRetrainedSVC": Optimize(
        LrcEmulationPipeline(
            LrcUllrich(
                **LrcUllrich.PredefinedParameters.untrained_svc
            ).set_params(clf_pipe__clf__C=1)
        )
    ),
    "UllrichRetrainedRF": Optimize(
        LrcEmulationPipeline(
            LrcUllrich(
                **LrcUllrich.PredefinedParameters.untrained_svc
            ).set_params(clf_pipe__clf=RandomForestClassifier())
        )
    ),
}

# %%
# Setting up the dataset
# ----------------------
# We run the comparison on the Lab and the Free-Living part of the TVS dataset.
# We use the :class:`~mobgap.data.TVSFreeLivingDataset` and the :class:`~mobgap.data.TVSLabDataset` to load the data.
# Note, that we use Memory caching to speed up the loading of the data.
# We also skip the recordings where the reference data is missing.
# In both cases, we compare against the INDIP reference system as done in the original validation as well.
#
# In the evaluation, each row of the dataset is treated as a separate recording.
# Results are calculated per recording.
# Aggregated results are calculated over the whole dataset, without considering the content of the individual
# recordings.
# Depending on how you want to interpret the results, you might not want to use the aggregated results, but rather
# perform custom aggregations over the provided "single_results".
from pathlib import Path

from joblib import Memory
from mobgap import PACKAGE_ROOT
from mobgap.data import TVSFreeLivingDataset, TVSLabDataset

cache_dir = Path(
    get_env_var("MOBGAP_CACHE_DIR_PATH", PACKAGE_ROOT.parent / ".cache")
)

datasets_free_living = TVSFreeLivingDataset(
    get_env_var("MOBGAP_TVS_DATASET_PATH"),
    reference_system="INDIP",
    memory=Memory(cache_dir),
    missing_reference_error_type="skip",
)
datasets_laboratory = TVSLabDataset(
    get_env_var("MOBGAP_TVS_DATASET_PATH"),
    reference_system="INDIP",
    memory=Memory(cache_dir),
    missing_reference_error_type="skip",
)


# %%
# Running the evaluation
# ----------------------
# We multiprocess the evaluation on the level of algorithms using joblib.
# Each algorithm pipeline is run using its own instance of the :class:`~mobgap.evaluation.Evaluation` class.
#
# The evaluation object iterates over the entire dataset, runs the algorithm on each recording and calculates the
# score using the :func:`~mobgap.initial_contacts._evaluation_scorer.icd_score` function.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from mobgap.laterality.evaluation import lrc_score
from mobgap.utils.evaluation import EvaluationCV
from tpcp.validate import DatasetSplitter

n_jobs = int(get_env_var("MOBGAP_N_JOBS", 3))
results_base_path = (
    Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH")) / "results/lrc_with_cv"
)


def run_evaluation(name, pipeline, ds):
    eval_pipe = EvaluationCV(
        ds,
        scoring=lrc_score,
        cv_iterator=DatasetSplitter(
            StratifiedGroupKFold(5), stratify="cohort", groupby="participant_id"
        ),
        cv_params={"n_jobs": n_jobs},
    ).run(pipeline)
    return name, eval_pipe


# %%
def eval_debug_plot(
    results: dict[str, EvaluationCV[LrcEmulationPipeline]],
) -> None:
    results_df = (
        pd.concat({k: v.get_single_results_as_df() for k, v in results.items()})
        .reset_index()
        .rename(columns={"level_0": "algo_name"})
    )

    sns.boxplot(
        data=results_df,
        x="cohort",
        y="accuracy",
        hue="algo_name",
        showmeans=True,
    )
    plt.tight_layout()
    plt.show()


# %%
# Free-Living
# ~~~~~~~~~~~
# Let's start with the Free-Living part of the dataset.

with Parallel(n_jobs=1) as parallel:
    results_free_living: dict[str, EvaluationCV[LrcEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_free_living)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
# This is not meant to be a comprehensive analysis, but rather a quick check to see if the results are as expected.
eval_debug_plot(results_free_living)

# %%
# Then we save the results to disk.
from mobgap.utils.evaluation import save_evaluation_results

for k, v in results_free_living.items():
    save_evaluation_results(
        k,
        v,
        condition="free_living",
        base_path=results_base_path,
        raw_result_filter=["detected"],
    )


# %%
# Laboratory
# ~~~~~~~~~~
# Now, we repeat the evaluation for the Laboratory part of the dataset.
with Parallel(n_jobs=n_jobs) as parallel:
    results_laboratory: dict[str, EvaluationCV[LrcEmulationPipeline]] = dict(
        parallel(
            delayed(run_evaluation)(name, pipeline, datasets_laboratory)
            for name, pipeline in pipelines.items()
        )
    )

# %%
# We create a quick plot for debugging.
eval_debug_plot(results_laboratory)

# %%
# Then we save the results to disk.
for k, v in results_laboratory.items():
    save_evaluation_results(
        k,
        v,
        condition="laboratory",
        base_path=results_base_path,
        raw_result_filter=["detected"],
    )
