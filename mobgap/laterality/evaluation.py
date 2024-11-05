from sklearn.metrics import accuracy_score
from tpcp.validate import NoAgg

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.laterality.pipeline import LrcEmulationPipeline




def laterality_evaluation_scorer(pipeline: LrcEmulationPipeline, datapoint: BaseGaitDatasetWithReference):
    predicted_lr_labels = pipeline.clone().safe_run(datapoint).ic_lr_list_

    ref_labels = datapoint.reference_parameters_.ic_list["lr_label"]

    combined = predicted_lr_labels.assign(ref_lr_label=ref_labels)
    # TODO: Wrap with Nan-Mean agg scorer
    return {"accuracy": accuracy_score(ref_labels, predicted_lr_labels["lr_label"]), "raw_results": NoAgg(combined)}
