"""Model factory class.

Return a model that matches input query.
"""
import logging

from gensim.models import KeyedVectors
from models import cross_net, memn2n, nn_utils, svm_mohammad17, transformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelFactory:
    """Generate ML models from query."""
    wordvec = None

    @classmethod
    def get_model(cls, query: str, wvfp: str, profile: bool=False, params: dict=None) -> Pipeline:
        # TODO: remove target arguments? now it's useless
        if wvfp and cls.wordvec is None:
            # load only once for efficiency
            cls.wordvec = KeyedVectors.load_word2vec_format(wvfp, binary=False)

        if query == 'svm':
            return svm_mohammad17.get_model(cls.wordvec, profile=profile)
        else:  # assume neural nets
            if query in ['crossnet', 'cn', 'CrossNet', 'crossNet']:
                build_fn = cross_net.build_model
            elif query in ['memnet', 'MemNet', 'mn', 'memNet', 'AttNet', 'attnet']:
                build_fn = memn2n.build_model
            elif query in ['tf', 'transformer', 'Transformer']:
                build_fn = transformer.build_model
            else:
                raise NotImplementedError(f'"{query}" is not implemented')

            if params is None:
                model = nn_utils.NeuralPipeline(build_fn,
                                                wordvec=cls.wordvec,
                                                profile=profile)
            else:
                model = nn_utils.NeuralPipeline(build_fn,
                                                wordvec=cls.wordvec,
                                                profile=profile,
                                                **params)

            logger.info('params: ' + repr(model.get_params()))
            return model
