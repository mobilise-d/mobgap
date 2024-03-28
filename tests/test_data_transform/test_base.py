import numpy as np
import pytest

from mobgap.data_transform import Resample
from mobgap.data_transform._utils import chain_transformers
from mobgap.data_transform.base import BaseTransformer


class TestChainTransformers:
    def test_chaining_works_in_order(self):
        class AddingTransformer(BaseTransformer):
            def transform(self, data, **kwargs):
                self.transformed_data_ = data + 1
                return self

        class MultiplyingTransformer(BaseTransformer):
            def transform(self, data, **kwargs):
                self.transformed_data_ = data * 2
                return self

        data = np.array([1, 2, 3])

        result = chain_transformers(data, [("add", AddingTransformer()), ("multiply", MultiplyingTransformer())])

        assert np.allclose(result, np.array([4, 6, 8]))

    @pytest.mark.parametrize("raise_in_chain", [0, 1, 2])
    def test_error_message_contains_transformer_name(self, raise_in_chain):
        class ErrorTransformer(BaseTransformer):
            def __init__(self, should_raise=False) -> None:
                self.should_raise = should_raise

            def transform(self, data, **kwargs):
                if self.should_raise:
                    raise RuntimeError("Error")
                self.transformed_data_ = data
                return self

        transformer_chain = [(str(i), ErrorTransformer()) for i in range(3)]

        transformer_chain[raise_in_chain][1].set_params(should_raise=True)

        with pytest.raises(RuntimeError) as e:
            chain_transformers(np.array([1, 2, 3]), transformer_chain)

        assert transformer_chain[raise_in_chain][0] in str(e.value)

    def test_kwargs_forwarded(self):
        class MultiplyBySamplingRate(BaseTransformer):
            def transform(self, data, *, sampling_rate_hz=None, **kwargs):
                self.transformed_data_ = data * sampling_rate_hz
                return self

        result = chain_transformers(
            np.ones(10),
            [("multiply1", MultiplyBySamplingRate()), ("multiply2", MultiplyBySamplingRate())],
            sampling_rate_hz=100.0,
        )

        assert np.all(result == 100.0**2)

    @pytest.mark.parametrize("sampling_rate", [100.0, 50.0])
    def test_chain_with_resample(self, sampling_rate):
        """Resample updates the sampling rate for all subsequent transformers."""

        class MultiplyBySamplingRate(BaseTransformer):
            def transform(self, data, *, sampling_rate_hz=None, **kwargs):
                self.transformed_data_ = data * sampling_rate_hz
                return self

        result = chain_transformers(
            np.ones(1000),
            [("resample", Resample(target_sampling_rate_hz=sampling_rate)), ("multiply", MultiplyBySamplingRate())],
            sampling_rate_hz=100.0,
        )

        assert np.all(result == 1 * sampling_rate)
