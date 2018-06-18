import unittest

import numpy as np

import data_utility as du


class TestGetX(unittest.TestCase):
    doc = {'company': 'adani', 'tweet_t': "#stopadani is negative stance for adani", 'profile_t': "here is description text #goadani"}

    def test_no_option(self):
        self.assertEqual(
            self.doc['company'] + '\t' + self.doc['tweet_t'] + '\t' + self.doc['profile_t'],
            du.get_x(self.doc, rm_autotag=False, profile=True)
        )

    def test_rm_autotag(self):
        self.assertEqual(
            self.doc['company'] + '\t' + " is negative stance for adani",
            du.get_x(self.doc, rm_autotag=True, profile=False)
        )

    def test_profile(self):
        self.assertEqual(
            self.doc['company'] + '\t' + self.doc['tweet_t'] + '\t' + self.doc['profile_t'],
            du.get_x(self.doc, rm_autotag=False, profile=True)
        )

    def test_autotag_profile(self):
        self.assertEqual(
            self.doc['company'] + '\t' + " is negative stance for adani" + '\t' + "here is description text ",
            du.get_x(self.doc, rm_autotag=True, profile=True)
        )


if __name__ == '__main__':
    unittest.main()
