"""
GSD TVS Evaluation
==================
This script shows the actual performance evaluation that is run to benchmark the GSD algorithms on the TVS dataset.
As this is run on the TVS dataset and has a long runtime, this example needs to be downloaded and run locally.
"""

import os
from pathlib import Path

from joblib import Memory
from mobgap import PACKAGE_ROOT
from mobgap.data import TVSFreeLivingDataset

if "MOBGAP_TVS_DATASET_PATH" not in os.environ:
    raise ValueError(
        "Please set the environmental variable MOBGAP_TVS_DATASET_PATH to the path of the TVS dataset."
    )

dataset_path = Path(os.getenv("MOBGAP_TVS_DATASET_PATH"))
n_jobs = os.environ.get("MOBGAP_N_JOBS", 1)

free_living_data = TVSFreeLivingDataset(
    dataset_path,
    reference_system="INDIP",
    memory=Memory(PACKAGE_ROOT.parent / ".cache"),
    missing_reference_error_type="skip",
).get_subset(recording="Recording4")

# %%
# Run the non-CV evaluation for all implemented algorithms
from mobgap.gait_sequences import GsdAdaptiveIonescu, GsdIluz, GsdIonescu
from mobgap.gait_sequences.pipeline import (
    GsdEmulationPipeline,
    GsdEvaluation,
    gsd_evaluation_scorer,
)

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
