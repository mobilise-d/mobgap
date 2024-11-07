import warnings

import pandas as pd
from joblib import Memory
from sklearn.model_selection import ParameterGrid
from tpcp.optimize import GridSearchCV
from tpcp.validate import validate

from mobgap import PACKAGE_ROOT
from mobgap.data import MsProjectDataset, TVSFreeLivingDataset, TVSLabDataset
from mobgap.laterality import LrcUllrich
from mobgap.laterality.evaluation import laterality_evaluation_scorer
from mobgap.laterality.pipeline import LrcEmulationPipeline
from mobgap.utils.misc import get_env_var

ms_project_dataset = MsProjectDataset(
    base_path=get_env_var("MOBGAP_MSPROJECT_DATASET_PATH"), reference_system="SU_LowerShanks"
)
tvs_free_living = TVSFreeLivingDataset(
    base_path=get_env_var("MOBGAP_TVS_DATASET_PATH"),
    reference_system="INDIP",
    memory=Memory(PACKAGE_ROOT / ".cache"),
    missing_reference_error_type="skip",
)
tvs_lab = TVSLabDataset(
    base_path=get_env_var("MOBGAP_TVS_DATASET_PATH"),
    reference_system="INDIP",
    memory=Memory(PACKAGE_ROOT / ".cache"),
    missing_reference_error_type="skip",
)


def scoring_no_warning(pipeline, datapoint):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="There were multiple ICs")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
        return laterality_evaluation_scorer(pipeline, datapoint)


# %%
# Old model
results = validate(
    LrcEmulationPipeline(LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all_old)),
    tvs_free_living,
    n_jobs=3,
    scoring=scoring_no_warning,
)
results = pd.DataFrame(results).drop(columns="single__raw_results")
print("Old results TVS FL", results["single__accuracy"].explode().agg(["mean", "median", "std"]))


# %%
# Current version of the new model
results = validate(
    LrcEmulationPipeline(LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all)),
    tvs_free_living,
    n_jobs=3,
    scoring=scoring_no_warning,
)
results = pd.DataFrame(results).drop(columns="single__raw_results")
print("Current results TVS RL", results["single__accuracy"].explode().agg(["mean", "median", "std"]))

# %%
# Single Full Retrain
# with Gridsearch
parameters = ParameterGrid({"algo__clf_pipe__clf__C": [0.01, 0.1, 1, 10, 100]})
algo = LrcUllrich(**LrcUllrich.PredefinedParameters.untrained_svc)

pipeline_trained_all = LrcEmulationPipeline(algo)
gs = GridSearchCV(
    pipeline_trained_all, parameters, n_jobs=3, scoring=scoring_no_warning, cv=5, return_optimized="accuracy"
)
gs.optimize(ms_project_dataset)

optimized_pipeline = gs.optimized_pipeline_

# NOTE: The new model has C = 1 as optimal parameter compared to the old model with C = 0.1
print("Optimized C", optimized_pipeline.algo.clf_pipe.named_steps["clf"].C)

# %%
results = validate(optimized_pipeline, tvs_free_living, n_jobs=3, scoring=scoring_no_warning)
results = pd.DataFrame(results).drop(columns="single__raw_results")
print("New results TVS", results["single__accuracy"].explode().agg(["mean", "median", "std"]))

# %%
# Save the model
import joblib

joblib.dump(
    optimized_pipeline.algo.clf_pipe, PACKAGE_ROOT / "laterality/_ullrich_pretrained_models/msproject_all_model.gz"
)
