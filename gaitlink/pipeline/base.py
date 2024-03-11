from tpcp import Pipeline

from gaitlink.data.base import BaseGaitlinkDataset


class BaseMobilisedDataPipeline(Pipeline[BaseGaitlinkDataset]):
    """Base class for pipelines that work with"""

    def