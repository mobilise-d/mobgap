# We only run some simple tests. For the rest the example regression tests should be sufficient.
import pytest
from tpcp.optimize import DummyOptimize
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.gait_sequences import GsdIluz
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline
from mobgap.utils.evaluation import Evaluation, EvaluationCV

short_example_data = LabExampleDataset().get_subset(test="Test5", trial="Trial1")


def dummy_scoring(x, y):
    return {"bla": 0}


class TestMetaEvaluationCV(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = EvaluationCV
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(short_example_data, cv_iterator=2, scoring=dummy_scoring).run(
            DummyOptimize(GsdEmulationPipeline(GsdIluz()), ignore_potential_user_error_warning=True)
        )


class TestMetaEvaluation(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = Evaluation
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(short_example_data, scoring=dummy_scoring).run(GsdEmulationPipeline(GsdIluz()))
