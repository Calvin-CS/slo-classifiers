"""
SLO Topic Modeling
Advisor: Professor VanderLinden
Name: Joseph Jinn
Date: 8-6-19
Versions: 1.0

SLO Topic Extraction Utility Functions Settings.

###########################################################
Notes:

Currently contains various constants and static data structures used in dataset pre-processing and post-processing in
preparation for topic extraction.

###########################################################
Resources Used:

Adapted from settings.py in slo-classifiers stance detection.

"""

################################################################################################################
################################################################################################################

# Import libraries.
import logging as log
import re
import warnings
import time
import pandas as pd

#############################################################

# Miscellaneous parameter adjustments for pandas and python.
# Pandas options.
pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = 1000
# Pandas float precision display.
pd.set_option('precision', 12)
# Don't output these types of warnings to terminal.
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

"""
Turn debug log statements for various sections of code on/off.
(adjust log level as necessary)
"""
log.basicConfig(level=log.INFO)
log.disable(level=log.DEBUG)

################################################################################################################
################################################################################################################

# Patterns for important tweet sub-strings
PTN_rt = re.compile(r'^(rt @\w+: )')
PTN_url = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
PTN_concatenated_url = re.compile(r'(.)http')
PTN_mention = re.compile(r'@[a-zA-Z_0-9]+')
PTN_stock_symbol = re.compile(r'\$[a-zA-Z]+')
PTN_hash = re.compile(r'#\w+')
PTN_whitespace = re.compile(r'\s+')
PTN_elongation = re.compile(r'(.)\1{2,}')
PTN_year = re.compile(r'[12][0-9]{3}')
PTN_time = re.compile(r'[012]?[0-9]:[0-5][0-9]')
# https://stackoverflow.com/a/13848829
PTN_cash = re.compile(
    r'\$(?=\(.*\)|[^()]*$)\(?\d{1,3}(,?\d{3})?(\.\d\d?)?\)?([bmk]| hundred| thousand| million| billion)?')


def process_tweet_text(tweet_text):
    """This function runs through the text regularization expressions to produce
    a processed tweet text."""

    # Remove "RT" tags.
    tweet_text = PTN_rt.sub("", tweet_text)
    # Remove concatenated URL's.
    tweet_text = PTN_concatenated_url.sub(r'\1 http', tweet_text)
    # Handle whitespaces.
    tweet_text = PTN_whitespace.sub(r' ', tweet_text)
    # Remove URL's.
    tweet_text = PTN_url.sub(r"slo_url", tweet_text)
    # Remove Tweet user mentions.
    tweet_text = PTN_mention.sub(r'slo_mention', tweet_text)
    # Remove Tweet stock symbols.
    tweet_text = PTN_stock_symbol.sub(r'slo_stock', tweet_text)
    # Remove Tweet hashtags.
    tweet_text = PTN_hash.sub(r'slo_hash', tweet_text)
    # Remove Tweet cashtags.
    tweet_text = PTN_cash.sub(r'slo_cash', tweet_text)
    # Remove Tweet year.
    tweet_text = PTN_year.sub(r'slo_year', tweet_text)
    # Remove Tweet time.
    tweet_text = PTN_time.sub(r'slo_time', tweet_text)
    # Remove character elongations.
    tweet_text = PTN_elongation.sub(r'\1\1\1', tweet_text)

    return tweet_text


# Specify tokens to remove during pre-processing function.
delete_list = []

# Specify tokens to remove during post-processing function. (from Derek Fisher's code + additions)
irrelevant_words = ["word_n", "auspol", "ausbiz", "tinto", "adelaide", "csg", "nswpol",
                    "nsw", "lng", "don", "rio", "pilliga", "australia", "asx", "just", "today", "great", "says", "like",
                    "big", "better", "rite", "would", "SCREEN_NAME", "mining", "former", "qldpod", "qldpol", "qld",
                    "wr",
                    "melbourne", "andrew", "fuck", "spadani", "greg", "th", "australians", "http", "https", "rt",
                    "co", "amp", "carmichael", "abbot", "bill shorten",
                    "slo_url", "slo_mention", "slo_hash", "slo_year", "slo_time", "slo_cash", "slo_stock",
                    "adani", "bhp", "cuesta", "fotescue", "riotinto", "newmontmining", "santos", "oilsearch",
                    "woodside", "ilukaresources", "whitehavencoal",
                    "stopadani", "goadani", "bhpbilliton", "billiton", "cuestacoal", "cuests coal", "cqc",
                    "fortescuenews", "fortescue metals", "rio tinto", "newmont", "newmont mining", "santosltd",
                    "oilsearchltd", "oil search", "woodsideenergy", "woodside petroleum", "woodside energy",
                    "iluka", "iluka resources", "whitehaven", "whitehaven coal"]

################################################################################################################

"""
Main function.
Unit tests for regular expressions.
"""
if __name__ == '__main__':
    start_time = time.time()

    # def whitespace_assertion_test(input_string, expected_string):
    #     """
    #     Unit test for removing extra whitespace characters.
    #     :param input_string: Tweet text to test.
    #     :param expected_string:  expected results to assert.
    #     :return:
    #     """
    #     preprocessed_tweet_text = PTN_whitespace.sub(r' ', input_string)
    #     print(f"Extra Whitespace Assertion Test Result: {preprocessed_tweet_text}")
    #     assert PTN_whitespace.sub(r' ', input_string) == expected_string


    def rt_assertion_test(input_string, expected_string):
        """
        Unit test for removing retweet tags.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_rt.sub("", input_string)
        # print(f"RT Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_rt.sub("", input_string) == expected_string


    def url_assertion_test(input_string, expected_string):
        """
        Unit test for removing URL's.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_url.sub(r"slo_url", input_string)
        # print(f"URL Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_url.sub("slo_url", input_string) == expected_string


    def concat_url_assertion_test(input_string, expected_string):
        """
        Unit test for removing concatenated URL's.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        FIXME - is this working as intended?  It seems to just add an extra whitespace.
        """
        preprocessed_tweet_text = PTN_concatenated_url.sub(r'\1 http', input_string)
        # print(f"Concatenated URL Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_concatenated_url.sub(r'\1 http', input_string) == expected_string


    def user_mentions_assertion_test(input_string, expected_string):
        """
        Unit test for removing user mentions.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_mention.sub(r'slo_mention', input_string)
        # print(f"User Mentions Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_mention.sub(r'slo_mention', input_string) == expected_string


    def stock_symbols_assertion_test(input_string, expected_string):
        """
        Unit test for removing stock symbols.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_stock_symbol.sub(r'slo_stock', input_string)
        # print(f"Stock Symbols Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_stock_symbol.sub(r'slo_stock', input_string) == expected_string


    def hashtags_assertion_test(input_string, expected_string):
        """
        Unit test for removing hashtags.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_hash.sub(r'slo_hash', input_string)
        # print(f"Hashtags Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_hash.sub(r'slo_hash', input_string) == expected_string


    def cashtags_assertion_test(input_string, expected_string):
        """
        Unit test for removing cashtags.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_cash.sub(r'slo_cash', input_string)
        # print(f"Cashtags Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_cash.sub(r'slo_cash', input_string) == expected_string


    def year_assertion_test(input_string, expected_string):
        """
        Unit test for removing year stamps.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_year.sub(r'slo_year', input_string)
        # print(f"Year Stamp Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_year.sub(r'slo_year', input_string) == expected_string


    def time_assertion_test(input_string, expected_string):
        """
        Unit test for removing time stamps.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_time.sub(r'slo_time', input_string)
        # print(f"Time Stamp Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_time.sub(r'slo_time', input_string) == expected_string


    def char_elongation_assertion_test(input_string, expected_string):
        """
        Unit test for removing character elongations.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = PTN_elongation.sub(r'\1\1\1', input_string)
        # print(f"Character Elongation Assertion Test Result: {preprocessed_tweet_text}")
        assert PTN_elongation.sub(r'\1\1\1', input_string) == expected_string


    def meta_assertion_test(input_string, expected_string):
        """
        Unit test for all regular expressions used in Tweet text pre-processing.
        :param input_string: Tweet text to test.
        :param expected_string:  expected results to assert.
        :return:
        """
        preprocessed_tweet_text = process_tweet_text(input_string)

        # print(f"Meta Assertion Test Result: {preprocessed_tweet_text}")
        assert preprocessed_tweet_text == expected_string


    def run_assertion_tests(pattern, replacement, test_cases):
        """This utility function runs the given list of pattern-replacement
        test cases.
        """
        for case in test_cases:
            assert pattern.sub(replacement, case["input"]) == case["expected"]


    ###########################################################

    whitespace_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": "  ",
            "expected": " "
        },
        {
            "input": " \t\n",
            "expected": " "
        },
        {
            "input": "RT @ozmining: News:  Adani applies for rail from Galilee to Abbot point http://t.co/CidyC4CkaM #mining",
            "expected": "RT @ozmining: News: Adani applies for rail from Galilee to Abbot point http://t.co/CidyC4CkaM #mining"
        }
    ]
    run_assertion_tests(PTN_whitespace, r' ', whitespace_assertion_test_cases)


    rt_assertion_test(
        "rt @ozmining: News:  Adani applies for rail from Galilee to Abbot point http://t.co/CidyC4CkaM #mining",
        "News:  Adani applies for rail from Galilee to Abbot point http://t.co/CidyC4CkaM #mining"
    )

    url_assertion_test(
        "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal",
        "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country slo_url @Johnsmccarthy #qldpol #coal"
    )

    concat_url_assertion_test(
        "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal",
        "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country  http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal"
    )

    user_mentions_assertion_test(
        "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal",
        "RT slo_mention: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J slo_mention #qldpol #coal"
    )

    stock_symbols_assertion_test(
        "WATCH all you need to know about today's #ASX drop. @ANZ_AU  $ANZ falls the least out of the #big4banks @netwealthInvest $NWL rises. @WoodsideEnergy  $WPL down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as $SDA trades at all time highs #Ausbiz https://t.co/TGWfUkhZi8",
        "WATCH all you need to know about today's #ASX drop. @ANZ_AU  slo_stock falls the least out of the #big4banks @netwealthInvest slo_stock rises. @WoodsideEnergy  slo_stock down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as slo_stock trades at all time highs #Ausbiz https://t.co/TGWfUkhZi8"
    )

    hashtags_assertion_test(
        "WATCH all you need to know about today's #ASX drop. @ANZ_AU  $ANZ falls the least out of the #big4banks @netwealthInvest $NWL rises. @WoodsideEnergy  $WPL down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as $SDA trades at all time highs #Ausbiz https://t.co/TGWfUkhZi8",
        "WATCH all you need to know about today's slo_hash drop. @ANZ_AU  $ANZ falls the least out of the slo_hash @netwealthInvest $NWL rises. @WoodsideEnergy  $WPL down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as $SDA trades at all time highs slo_hash https://t.co/TGWfUkhZi8"
    )

    cashtags_assertion_test(
        "RT @Johnsmccarthy: Adani signs $1b MOU with Bank of India for $16.5b Carmichael project as tipped in today's @couriermail",
        "RT @Johnsmccarthy: Adani signs slo_cash MOU with Bank of India for slo_cash Carmichael project as tipped in today's @couriermail"
    )

    year_assertion_test(
        "Adani Ports net profit up by 68pct in Q2 2014 - http://t.co/nyErocJxGd #GoogleAlerts",
        "Adani Ports net profit up by 68pct in Q2 slo_year - http://t.co/nyErocJxGd #GoogleAlerts"
    )

    time_assertion_test(
        "August 22, 2015 12:00AM  Greens delay Adani until 2017  Indian energy giant Adaniâ€™s $16 billion Carmichael mine... http://t.co/n429QU4w0D",
        "August 22, 2015 slo_timeAM  Greens delay Adani until 2017  Indian energy giant Adaniâ€™s $16 billion Carmichael mine... http://t.co/n429QU4w0D"
    )

    char_elongation_assertion_test(
        "Yaaaay stop the Adani Destruction Machine and Adani's mate Hunt https://t.co/U7zNlvYArn",
        "Yaaay stop the Adani Destruction Machine and Adani's mate Hunt https://t.co/U7zNlvYArn"
    )

    char_elongation_assertion_test(
        "AAAAAAAAAH  (I'm still filthy about Adani but this helps somewhat) https://t.co/sN84tXZqLX",
        "AAAH  (I'm still filthy about Adani but this helps somewhat) https://t.co/sN84tXZqLX"
    )

    full_tweet_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": "hello, tweet text!",
            "expected": "hello, tweet text!"
        },
        {
            "input": "WATCH all you need to know about today's #ASX drop. @ANZ_AU  $ANZ falls the least out of the #big4banks @netwealthInvest $NWL rises. @WoodsideEnergy  $WPL down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as $SDA trades at all time highs #Ausbiz https://t.co/TGWfUkhZi8",
            "expected": "WATCH all you need to know about today's slo_hash drop. slo_mention slo_stock falls the least out of the slo_hash slo_mention slo_stock rises. slo_mention slo_stock down 1.5% &amp; slo_mention trades lower amid profit taking as slo_stock trades at all time highs slo_hash slo_url"
        }
    ]
    for case in full_tweet_assertion_test_cases:
        assert process_tweet_text(case["input"]) == case["expected"]


    ###########################################################

    end_time = time.time()
    time_elapsed_seconds = end_time - start_time
    time_elapsed_minutes = (end_time - start_time) / 60.0
    time_elapsed_hours = (end_time - start_time) / 60.0 / 60.0
    time.sleep(3)
    log.info(f"tests passed...\n\ttime: {time_elapsed_seconds} seconds, "
             f"{time_elapsed_minutes} minutes, {time_elapsed_hours} hours")

    ################################################################################################################
