"""Test neural model API."""
import inspect
import unittest

from models import cross_net, memn2n, nn_utils, transformer


class TestNeuralPipeline(unittest.TestCase):
    kwargs_np = set(inspect.signature(nn_utils.NeuralPipeline.__init__).parameters.keys()) | {'dim_output', 'embedding_matrix', 'word_index'}

    def assertParamsCoverage(self, kwargs):
        for kwarg in kwargs:
            self.assertIn(kwarg, self.kwargs_np, msg=f'"{kwarg}" is not defined in NeuralPipeline.__init__')

    def _get_kwargs_from_func(self, func):
        sig = inspect.signature(func)
        return set(sig.parameters.keys())

    def test_params_coverage_crossnet(self):
        """Check all of the parameters of CrossNet are defined in NeuralPipeline's constructor."""
        self.longMessage = False
        kwargs_cn = self._get_kwargs_from_func(cross_net.build_model)
        self.assertParamsCoverage(kwargs_cn)

    def test_params_coverage_memnet(self):
        self.longMessage = False
        kwargs_mn = self._get_kwargs_from_func(memn2n.build_model)
        self.assertParamsCoverage(kwargs_mn)

    def test_params_coverage_transformer(self):
        self.longMessage = False
        kwargs_tf = self._get_kwargs_from_func(transformer.build_model)
        self.assertParamsCoverage(kwargs_tf)


if __name__ == '__main__':
    unittest.main()
