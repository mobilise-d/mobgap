import numpy as np
import pytest

from gaitlink.data_transform.base import BaseTransformer, chain_transformers


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
