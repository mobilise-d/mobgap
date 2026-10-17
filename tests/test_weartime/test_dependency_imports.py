"""Public wear-time dependency boundaries in fresh Python processes."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "required, blocked, code",
    [
        (
            (),
            ("tensorflow", "keras", "xgboost", "optuna"),
            "from mobgap.weartime import WtdMegaritisSignal, WearTimeModelConfig, TrainingData\n"
            "from mobgap.weartime.pipeline import WtdEmulationPipeline\n"
            "from mobgap.weartime.evaluation import wtd_score\nWtdMegaritisSignal()",
        ),
        (
            ("optuna",),
            ("tensorflow", "keras", "xgboost"),
            "from mobgap.weartime.optimization import WearTimeOptunaSearch",
        ),
        (
            ("xgboost", "optuna"),
            ("tensorflow", "keras"),
            "from mobgap.weartime.xgboost import WtdMegaritisXGBoost, WtdMegaritisXGBoostTrainer\n"
            "from mobgap.weartime.optimization import WearTimeOptunaSearch\n"
            "trainer = WtdMegaritisXGBoostTrainer(WtdMegaritisXGBoost())\n"
            "WearTimeOptunaSearch(trainer, "
            "**WtdMegaritisXGBoostTrainer.HyperparameterSearchPresets.sustain_weartime).clone().create_objective()",
        ),
        (
            ("tensorflow", "optuna"),
            ("xgboost",),
            "from mobgap.weartime.cnn import WtdMegaritisCNNTrainer\n"
            "from mobgap.weartime.optimization import WearTimeOptunaSearch\n"
            "trainer = WtdMegaritisCNNTrainer(**WtdMegaritisCNNTrainer.PredefinedParameters.empty_cnn)\n"
            "WearTimeOptunaSearch(trainer, "
            "**WtdMegaritisCNNTrainer.HyperparameterSearchPresets.sustain_weartime).clone().create_objective()",
        ),
        (
            (),
            ("tensorflow", "keras"),
            "try:\n import mobgap.weartime.cnn\n"
            "except ModuleNotFoundError as exc:\n assert exc.name == 'tensorflow'\n"
            "else:\n raise AssertionError('CNN import must require TensorFlow')",
        ),
        (
            (),
            ("xgboost",),
            "try:\n import mobgap.weartime.xgboost\n"
            "except ModuleNotFoundError as exc:\n assert exc.name == 'xgboost'\n"
            "else:\n raise AssertionError('XGBoost import must require XGBoost')",
        ),
    ],
    ids=["core", "optimizer", "xgboost-only", "cnn-only", "missing-tensorflow", "missing-xgboost"],
)
def test_public_imports_require_only_their_dependencies(required, blocked, code):
    """Unrelated optional dependencies do not prevent imports or optimizer construction."""
    for dependency in required:
        pytest.importorskip(dependency)
    setup = f"""import importlib.abc, sys
blocked = {blocked!r}
class MissingDependency(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path, target=None):
  if fullname.split('.')[0] in blocked:
   raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
sys.meta_path.insert(0, MissingDependency())
"""
    result = subprocess.run([sys.executable, "-c", setup + code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
