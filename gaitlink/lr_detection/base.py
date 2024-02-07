"""Base class for LR detectors"""

from typing import Any
from tpcp import Algorithm
import pandas as pd
from typing_extensions import Self, Unpack
from enum import StrEnum

import os
from pathlib import Path
import joblib

from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier



from gaitlink._docutils import make_filldoc


# I suggest this should be filled by an AI, to be consistent with the other algos...

base_lr_detector_docfiller = make_filldoc(
    {
        "other_parameters":
            """
        data
            """,
    },
    doc_summary = "Decorator to fill common parts of the docstring for subclasses of :class: `BaseLRDetector`.",
)

@base_lr_detector_docfiller
class BaseLRDetector(Algorithm):
    """
    Base class for LR detectors.
    
    This base class should be used for all Left/Right foot detection algorithms.
    
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.
    """
    _action_methods = ("detect",)
    
    # other_parameters
    imu_data: pd.DataFrame
    ic_list: list
    reference_data: bool
    
    # results
    LR_list: pd.DataFrame
    
    
    # TODO: presumably, the sampling rate should be inherited?
    
    @base_lr_detector_docfiller
    def detect(self, 
               imu_data: pd.DataFrame,
               ic_list: pd.Series,
               **kwargs: Any
               ) -> Self:
        """
        Add docs here.
        """
        raise NotImplementedError


# Deprecated
# Note that these models will need to be imported according to the groups of individuals they were trained on: HC, PD, MS, etc.
@base_lr_detector_docfiller
class PretrainedModel(StrEnum):
    """
    Enum class for the pretrained models
    """
    svm_linear = "svm_linear"
    svm_rbf = "svm_rbf"
    knn = "knn"
    rfc = "rfc"

    @staticmethod
    def load_pretrained_model(model_name):
        if model_name == PretrainedModel.svm_linear:
            base_dir = Path(os.getenv('MODEL_DIR', './pretrained_models'))
            model_path = base_dir / 'msproject_ms_model.gz'
            return joblib.load(model_path)
        
        # Note, these are not pretrained models, they are just some hyperparameters. They are here for convenience, since they were not available.
        elif model_name == "svm_rbf":
            return svm.SVC(kernel='rbf',
                           C=100, gamma=0.001,
                           probability=True)
            
        elif model_name == PretrainedModel.knn:
            return neighbors.KNeighborsClassifier(n_neighbors = 5)
        
        elif model_name == PretrainedModel.rfc:
            return RandomForestClassifier(n_estimators = 100,
                                          max_depth = 2,
                                          random_state = 0)
        else:
            raise NotImplementedError("The model specified is not supported yet.")
        

__all__ = ["BaseLRDetector", "base_lr_detector_docfiller", "PretrainedModel"]
    