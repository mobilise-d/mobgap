from joblib import Memory
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from tpcp.optimize import GridSearchCV
from tpcp.validate import DatasetSplitter

from mobgap import PROJECT_ROOT
from mobgap.data import MsProjectDataset
from mobgap.laterality import LrcUllrich
from mobgap.laterality.evaluation import lrc_score
from mobgap.laterality.pipeline import LrcEmulationPipeline
from mobgap.utils.misc import get_env_var

ms_project_dataset = MsProjectDataset(
    base_path=get_env_var("MOBGAP_MSPROJECT_DATASET_PATH"),
    reference_system="SU_LowerShanks",
    memory=Memory(PROJECT_ROOT / ".cache"),
)

# %%
# Single Full Retrain
# with Gridsearch
parameters = ParameterGrid({"algo__clf_pipe__clf__C": [0.01, 0.1, 1, 10, 100]})
algo = LrcUllrich(**LrcUllrich.PredefinedParameters.untrained_svc)

pipeline_trained_all = LrcEmulationPipeline(algo)
gs = GridSearchCV(
    pipeline_trained_all,
    parameters,
    n_jobs=3,
    scoring=lrc_score,
    cv=DatasetSplitter(StratifiedKFold(3), stratify="cohort"),
    return_optimized="accuracy",
)
gs.optimize(ms_project_dataset)

optimized_pipeline = gs.optimized_pipeline_

print("Optimized C", optimized_pipeline.algo.clf_pipe.named_steps["clf"].C)


# %%
# Save the model
import joblib

joblib.dump(
    optimized_pipeline.algo.clf_pipe, PROJECT_ROOT / "laterality/_ullrich_pretrained_models/msproject_all_model.gz"
)
