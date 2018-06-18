import unittest

from data import tweet_preprocessor as tp


class TestPreprocessing(unittest.TestCase):
    t = 'RT @mickb1611: Another blue marlin caught and successfully released yesterday.He put up a great fight and will live to fight another day #sologamefishing#garmin https://t.co/oVjA9R6cqC'

    def setUp(self):
        # to view the result of longer texts
        self.maxDiff = None

    def _test_base(self, challenge, response):
        self.assertEqual(response, tp.preprocess_text(challenge))

    def test_rt(self):
        self._test_base(
            'RT @mickb: another blue marlin caught and successfully released yesterday.',
            'another blue marlin caught and successfully released yesterday.'
        )

    def test_elongation(self):
        self._test_base(
            'savvyyyyy and great!!!!!! vvv',
            'savvyyy and great!!! vvv'
        )
        self._test_base(
            '@savvyyyyy and great!!!!!! vvv',
            '@savvyyyyy and great!!! vvv'
        )

    def test_html(self):
        self._test_base(
            'this &amp; that',
            'this & that'
        )

    def test_url(self):
        """check http concatenation and case preservation"""
        self._test_base(
            'released yesterday.He put up a great fight and will live to fight another day.https://t.co/oVjA9R6cqC',
            'released yesterday.he put up a great fight and will live to fight another day. https://t.co/oVjA9R6cqC'
        )

    def test_mention(self):
        self._test_base(
            '@miCKb1611 another blue marlin caught and successfully released yesterday @ island @1194 @hogehoge',
            '@miCKb1611 another blue marlin caught and successfully released yesterday @ island @1194 @hogehoge'
        )

    def test_numerals(self):
        self._test_base(
            '12:34 (time) 04:99 (not time) 25:32 (time) 3:03 1:2 :98 1968 2016 101 $12 $64,000 $5.1b $6 billion $\LaTeX$ 12$ $12, pay $1.',
            'slo_time (time) 04:99 (not time) slo_time (time) slo_time 1:2 :98 slo_year slo_year 101 slo_cash slo_cash slo_cash slo_cash $\latex$ 12$ slo_cash, pay slo_cash.'
        )


class TestPostprocessing(unittest.TestCase):

    def _test_base(self, challenge, response):
        self.assertEqual(tp.postprocess_text(challenge), response)

    def test_mention(self):
        self._test_base(
            '@miCKb1611 another blue marlin caught and successfully released yesterday @ island @1194 @hogehoge',
            'slo_mention another blue marlin caught and successfully released yesterday @ island slo_mention slo_mention'
        )

    def test_url(self):
        self._test_base(
            'released yesterday.he put up a great fight and will live to fight another day. https://t.co/oVjA9R6cqC',
            'released yesterday.he put up a great fight and will live to fight another day. slo_url'
        )


if __name__ == '__main__':
    unittest.main()
