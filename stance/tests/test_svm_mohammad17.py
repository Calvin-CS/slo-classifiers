import unittest

import numpy as np

import models.svm_mohammad17 as svm
from gensim.models import KeyedVectors

# WVFP = '../../../data/stance/wordvec/semeval_100d_ws10_min2.vec'
WVFP = 'dummy_wordvec.vec'


class TestTargetVectorizer(unittest.TestCase):
    def setUp(self):
        self.tv = svm.TargetVectorizer('Hillary Clinton')

    def test_transform(self):
        x = np.array([
            'clinton\tTrump said "make America great again"\tprofile',
            'hillary\thow are you, hillary\tprofile',
            'clinton\tClint east wood\tprofile',
            'clinton\tCLINTON and hiLLary\tprofile',  # check case insensitivity
            'clinton\tPresident #Clinton\tprofile',  # check Hashtags
        ])
        X = self.tv.transform(x)
        np.testing.assert_array_equal(
            X,
            np.array([[0], [1], [0], [1], [1]])
        )


class TestEmbeddingVectorizer(unittest.TestCase):
    def setUp(self):
        wordvec = KeyedVectors.load_word2vec_format(WVFP, binary=False)
        self.ev = svm.EmbeddingVectorizer(wordvec, profile=False)

    def test_transform_dim(self):
        x = np.array([
            'target\tmake America great again',
            'target\thow are you',
        ])
        X = self.ev.transform(x)
        # print(X)
        self.assertEqual(X.shape, (len(x), self.ev.wordvec_dim))

    def test_transform_unk(self):
        """Test return values are zeros vec when input values consist solely of unkown words."""
        x = np.array([
            'target\thoge hoge'
        ])
        X = self.ev.transform(x)
        np.testing.assert_array_equal(
            X,
            np.zeros((len(x), self.ev.wordvec_dim))
        )


class TestSLO_WordAnalyzer(unittest.TestCase):
    def setUp(self):
        self.slo_word_analyzer = svm.SLO_WordAnalyzer(False)
        self.slo_word_analyzer_p = svm.SLO_WordAnalyzer(True)

    def test_word_split(self):
        self.assertEqual(
            ['i', 'am', 'sure', '!'],
            self.slo_word_analyzer("i am sure !")
        )

    def test_profile(self):
        self.assertEqual(
            ['t_target', 'i', 'am', 'sure', '!', 'p_high', 'p_school', 'p_student', 'p_.'],
            self.slo_word_analyzer_p("target\ti am sure !\thigh school student ."),
        )


# class TestSVM(unittest.TestCase):
#     def __init__(self, arg):
#         pass


if __name__ == '__main__':
    unittest.main()
