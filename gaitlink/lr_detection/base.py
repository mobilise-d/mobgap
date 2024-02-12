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
        "detect_":
            """
        Method used for detection of left and right steps in the provided gait data. Predictions are stored post-execution using the `ic_lr` attribute of the instance, i.e. self.ic_lr.

        Parameters:
        ---------------
        data : pd.DataFrame
            The gait data.
        ic_list: pd.DataFrame
            The initial contact list, zero-indexed relative to the start of the GS.
        sampling_rate_hz : float
            The sampling rate in Hz.

        Returns:
        ---------------
        self: 
            The instance of the class.

        Predictions can be retrieved using self.ic_lr
            """,
    },
    doc_summary = "Decorator to fill common parts of the docstring for subclasses of :class: `BaseLRDetector`.",
)

@base_lr_detector_docfiller
class BaseLRDetector(Algorithm):
    """
    Base class for L/R foot detectors.
    
    This base class should be used for all Left/Right foot detection algorithms.
    
    Algorithms should implement the ``detect`` method, which will perform all relevant processing steps.

    The method should then return the instance of the class, with the ``ic_lr`` attribute set corresponding to the provided gait sequences.
 
    """
    _action_methods = ("detect",)
    
    # other_parameters
    data: list[pd.DataFrame]
    ic_list: list[pd.DataFrame]
    label_list: list[pd.DataFrame]
    sampling_rate_hz: float
    
    # results
    ic_lr: list[pd.DataFrame]
    
        
    @base_lr_detector_docfiller
    def detect(self, 
               data: list[pd.DataFrame],
               ic_list: list[pd.DataFrame],
               sampling_rate_hz: float,
               **kwargs: Unpack[dict[str, Any]]
               ) -> Self:
        """
        Add docs here.
        """
        raise NotImplementedError
        

__all__ = ["BaseLRDetector", "base_lr_detector_docfiller"]
    