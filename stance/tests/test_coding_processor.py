import unittest

from data import coding_processor as cp


class TestDatasetProcessor(unittest.TestCase):

    def test_tweet_existence(self):
        for tweet in [
            [847799312876576768, 'Liam_Wagner', True],    # ok
            # [916640818303090688, 'Jansant', False],       # suspended account (this one has been re-instated)
            [925552710912376832, 'abc730', False],        # non-existent tweet
            [804949582484340736, 'SeanBradbery', False],  # bad URL
        ]:
            assert (cp.check_tweet_accessibility(
                cp.create_tweet_url(tweet[0])
            ) == tweet[2])


if __name__ == '__main__':
    unittest.main()