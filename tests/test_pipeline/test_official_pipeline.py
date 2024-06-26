import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.pipeline import (
    GenericMobilisedPipeline,
    MobilisedMetaPipeline,
    MobilisedPipelineHealthy,
    MobilisedPipelineImpaired,
)


class TestMetaBaseMobilisedPipeline(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = GenericMobilisedPipeline
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(**self.ALGORITHM_CLASS.PredefinedParameters.regular_walking).run(
            LabExampleDataset()[0]
        )


class TestMetaMobilisedPipelineHealth(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedPipelineHealthy

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().run(LabExampleDataset()[0])


class TestMetaMobilisedPipelineImpaired(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedPipelineImpaired

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().run(LabExampleDataset()[0])


class TestMetaMobilisedMetaPipeline(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedMetaPipeline

    @pytest.fixture()
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().run(LabExampleDataset()[0])
