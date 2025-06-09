"""
GSD TVS Evaluation
==================
This script shows the actual performance evaluation that is run to benchmark the GSD algorithms on the TVS dataset.
As this is run on the TVS dataset and has a long runtime, this example needs to be downloaded and run locally.
"""

from pathlib import Path

from joblib import Memory
from mobgap import PROJECT_ROOT
from mobgap.data import TVSFreeLivingDataset
from mobgap.utils.misc import get_env_var

dataset_path = Path(get_env_var("MOBGAP_TVS_DATASET_PATH").strip('"'))
n_jobs = get_env_var("MOBGAP_N_JOBS", 1)

free_living_data = TVSFreeLivingDataset(
    dataset_path,
    reference_system="INDIP",
    memory=Memory(PROJECT_ROOT / ".cache"),
    missing_reference_error_type="skip",
).get_subset(recording="Recording4")

# %%
# Run the non-CV evaluation for all implemented algorithms
from mobgap.gait_sequences import GsdAdaptiveIonescu, GsdIluz, GsdIonescu
from mobgap.gait_sequences.evaluation import (
    GsdEvaluation,
    gsd_evaluation_scorer,
)
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline

results = {}
for algo in (GsdIluz(), GsdIonescu(), GsdAdaptiveIonescu()):
    print(f"Running evaluation for {algo.__class__.__name__}")
    pipe = GsdEmulationPipeline(algo)
    eval_pipe = GsdEvaluation(
        free_living_data,
        scoring=gsd_evaluation_scorer,
        validate_paras={"n_jobs": n_jobs},
    ).run(pipe)
    results[algo.__class__.__name__] = eval_pipe.results_
