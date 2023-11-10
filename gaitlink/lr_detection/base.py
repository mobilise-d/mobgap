"""Base class for LR detectors"""

from typing import Any
from tpcp import Algorithm
import pandas as pd
from typing_extensions import Self, Unpack
from gaitlink._docutils import make_filldoc

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
    
    Algorithms should implement the ``predict`` method, which will perform all relevant processing steps.
    """
    _action_methods = ("detect",)
    
    # other_parameters
    imu_data: pd.DataFrame
    event_list: list
    reference_data: bool
    
    # results
    LR_list: list
    
    
    # TODO: presumably, the sampling rate should be inherited?
    
    @base_lr_detector_docfiller
    def predict(self, 
               imu_data: pd.DataFrame,
               event_list: list,
               reference_data: bool,
               **kwargs: Any
               ) -> Self:
        """
        Add docs here.
        """
        raise NotImplementedError

__all__ = ["BaseLRDetector", "base_lr_detector_docfiller"]
    