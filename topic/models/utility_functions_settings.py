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

Also includes unit tests for each regular expression used in pre-processing of Tweet text.

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


def process_tweet_text(tweet_text):
    # TODO - re-arrange to be the exact same order as in tweet_preprocessor.py!
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


################################################################################################################

"""
Main function.
Unit tests for regular expressions.
"""
if __name__ == '__main__':
    start_time = time.time()


    def run_assertion_tests(pattern, replacement, test_cases):
        """This utility function runs the given list of pattern-replacement
        test cases.
        """
        counter = 0
        for _case in test_cases:
            print(f"Pattern: {str(pattern)} - Replacement: \"{str(replacement)}\" - "
                  f"\nTest Case: {str(_case)} - "
                  f"\nResult: {pattern.sub(replacement, _case['input'])}\n")
            assert pattern.sub(replacement, _case["input"]) == _case["expected"]
            counter += 1


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

    ###########################################################

    retweet_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "rt",
            "expected": "rt"
        },
        {
            "input": "RT",
            "expected": "RT"
        },
        {
            "input": "rt @:",
            "expected": "rt @:"
        },
        {
            "input": "rt @: ",
            "expected": "rt @: "
        },
        {
            "input": "rt @w: ",
            "expected": ""
        },
        {
            "input": "doesn't start with rt @w: ",
            "expected": "doesn't start with rt @w: "
        },
        {
            "input": "rt @ozmining: News:  Adani applies for rail from Galilee to Abbot point http://t.co/CidyC4CkaM #mining",
            "expected": "News:  Adani applies for rail from Galilee to Abbot point http://t.co/CidyC4CkaM #mining"
        }
    ]
    run_assertion_tests(PTN_rt, "", retweet_assertion_test_cases)

    ###########################################################

    url_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "http:",
            "expected": "http:"
        },
        {
            "input": "https:",
            "expected": "https:"
        },
        {
            "input": "http://",
            "expected": "http://"
        },
        {
            "input": "https://",
            "expected": "https://"
        },
        {
            "input": "http://h",
            "expected": "slo_url"
        },
        {
            "input": "https://h",
            "expected": "slo_url"
        },
        {
            "input": "http://hello_world",
            "expected": "slo_url"
        },
        {
            "input": "https://hello_world",
            "expected": "slo_url"
        },
        {
            "input": "https://hello_world.com",
            "expected": "slo_url"
        },
        {
            "input": "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal",
            "expected": "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country slo_url @Johnsmccarthy #qldpol #coal"
        }
    ]
    run_assertion_tests(PTN_url, r'slo_url', url_assertion_test_cases)

    ###########################################################

    concat_url_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "ahttp",
            "expected": "a http"
        },
        {
            "input": "aaaahttp",
            "expected": "aaaa http"
        },
        {
            "input": "aaaahttps",
            "expected": "aaaa https"
        },
        {
            "input": "aaaahttp://hello_world.com",
            "expected": "aaaa http://hello_world.com"
        },
        {
            "input": "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal",
            "expected": "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country  http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal"
        }
    ]
    run_assertion_tests(PTN_concatenated_url, r'\1 http', concat_url_assertion_test_cases)

    ###########################################################

    user_mentions_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "@",
            "expected": "@"
        },
        {
            "input": "@m",
            "expected": "slo_mention"
        },
        {
            "input": "@_m",
            "expected": "slo_mention"
        },
        {
            "input": "@0_m",
            "expected": "slo_mention"
        },
        {
            "input": "@mention",
            "expected": "slo_mention"
        },
        {
            "input": "mention@mention",
            "expected": "mentionslo_mention"
        },
        {
            "input": "mention @mention",
            "expected": "mention slo_mention"
        },
        {
            "input": "RT @peaceofwild: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J @Johnsmccarthy #qldpol #coal",
            "expected": "RT slo_mention: Absolutely flabbergasted. Adani proposing ANOTHER railline through CQ grazing country http://t.co/TpxYZZwM7J slo_mention #qldpol #coal"
        }
    ]
    run_assertion_tests(PTN_mention, r'slo_mention', user_mentions_assertion_test_cases)

    ###########################################################

    stock_symbols_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "$",
            "expected": "$"
        },
        {
            "input": "$s",
            "expected": "slo_stock"
        },
        {
            "input": "@_m",
            "expected": "@_m"
        },
        {
            "input": "@0_m",
            "expected": "@0_m"
        },
        {
            "input": "$stock",
            "expected": "slo_stock"
        },
        {
            "input": "stock$stock",
            "expected": "stockslo_stock"
        },
        {
            "input": "stock $stock",
            "expected": "stock slo_stock"
        },
        {
            "input": "WATCH all you need to know about today's #ASX drop. @ANZ_AU  $ANZ falls the least out of the #big4banks @netwealthInvest $NWL rises. @WoodsideEnergy  $WPL down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as $SDA trades at all time highs #Ausbiz https://t.co/TGWfUkhZi8",
            "expected": "WATCH all you need to know about today's #ASX drop. @ANZ_AU  slo_stock falls the least out of the #big4banks @netwealthInvest slo_stock rises. @WoodsideEnergy  slo_stock down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as slo_stock trades at all time highs #Ausbiz https://t.co/TGWfUkhZi8"
        }
    ]
    run_assertion_tests(PTN_stock_symbol, r'slo_stock', stock_symbols_assertion_test_cases)

    ###########################################################

    hashtags_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "#",
            "expected": "#"
        },
        {
            "input": "#h",
            "expected": "slo_hash"
        },
        {
            "input": "#_h",
            "expected": "slo_hash"
        },
        {
            "input": "#0_h",
            "expected": "slo_hash"
        },
        {
            "input": "#hash",
            "expected": "slo_hash"
        },
        {
            "input": "hash#hash",
            "expected": "hashslo_hash"
        },
        {
            "input": "hash #hash",
            "expected": "hash slo_hash"
        },
        {
            "input": "WATCH all you need to know about today's #ASX drop. @ANZ_AU  $ANZ falls the least out of the #big4banks @netwealthInvest $NWL rises. @WoodsideEnergy  $WPL down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as $SDA trades at all time highs #Ausbiz https://t.co/TGWfUkhZi8",
            "expected": "WATCH all you need to know about today's slo_hash drop. @ANZ_AU  $ANZ falls the least out of the slo_hash @netwealthInvest $NWL rises. @WoodsideEnergy  $WPL down 1.5% &amp; @Speedcast_Intl  trades lower amid profit taking as $SDA trades at all time highs slo_hash https://t.co/TGWfUkhZi8"
        }
    ]
    run_assertion_tests(PTN_hash, r'slo_hash', hashtags_assertion_test_cases)

    ###########################################################

    # This one does interesting things depending on # of digits.
    cashtags_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "$",
            "expected": "$"
        },
        {
            "input": "$a",
            "expected": "$a"
        },
        {
            "input": "$0bmk",
            "expected": "slo_cashmk"
        },
        {
            "input": "$0thousand",
            "expected": "slo_cashthousand"
        },
        {
            "input": "$0million",
            "expected": "slo_cashillion"
        },
        {
            "input": "$0billion",
            "expected": "slo_cashillion"
        },
        {
            "input": "$1",
            "expected": "slo_cash"
        },
        {
            "input": "$1.",
            "expected": "slo_cash."
        },
        {
            "input": "$1.1",
            "expected": "slo_cash"
        },
        {
            "input": "$11.11",
            "expected": "slo_cash"
        },
        {
            "input": "$1111.11",
            "expected": "slo_cash1.11"
        },
        {
            "input": "$11.1111",
            "expected": "slo_cash11"
        },
        {
            "input": "$1234",
            "expected": "slo_cash4"
        },
        {
            "input": "$123456",
            "expected": "slo_cash"
        },
        {
            "input": "$12345678",
            "expected": "slo_cash78"
        },
        {
            "input": "$1234567890",
            "expected": "slo_cash7890"
        },
        {
            "input": "RT @Johnsmccarthy: Adani signs $1b MOU with Bank of India for $16.5b Carmichael project as tipped in today's @couriermail",
            "expected": "RT @Johnsmccarthy: Adani signs slo_cash MOU with Bank of India for slo_cash Carmichael project as tipped in today's @couriermail"
        }
    ]
    run_assertion_tests(PTN_cash, r'slo_cash', cashtags_assertion_test_cases)

    ###########################################################

    year_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "0986",
            "expected": "0986"
        },
        {
            "input": "1986",
            "expected": "slo_year"
        },
        {
            "input": "2986",
            "expected": "slo_year"
        },
        {
            "input": "3986",
            "expected": "3986"
        },
        {
            "input": "01986",
            "expected": "0slo_year"
        },
        {
            "input": "19863",
            "expected": "slo_year3"
        },
        {
            "input": "1986a",
            "expected": "slo_yeara"
        },
        {
            "input": "a1986a",
            "expected": "aslo_yeara"
        },
        {
            "input": "Adani Ports net profit up by 68pct in Q2 2014 - http://t.co/nyErocJxGd #GoogleAlerts",
            "expected": "Adani Ports net profit up by 68pct in Q2 slo_year - http://t.co/nyErocJxGd #GoogleAlerts"
        }
    ]
    run_assertion_tests(PTN_year, r'slo_year', year_assertion_test_cases)

    ###########################################################

    time_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "00:00",
            "expected": "slo_time"
        },
        {
            "input": "10:00",
            "expected": "slo_time"
        },
        {
            "input": "20:00",
            "expected": "slo_time"
        },
        {
            "input": "30:00",
            "expected": "3slo_time"
        },
        {
            "input": "a00:00",
            "expected": "aslo_time"
        },
        {
            "input": "a0:00",
            "expected": "aslo_time"
        },
        {
            "input": "0:00",
            "expected": "slo_time"
        },
        {
            "input": "a00:00a",
            "expected": "aslo_timea"
        },
        {
            "input": "a0:0a",
            "expected": "a0:0a"
        },
        {
            "input": "August 22, 2015 12:00AM  Greens delay Adani until 2017  Indian energy giant Adaniâ€™s $16 billion Carmichael mine... http://t.co/n429QU4w0D",
            "expected": "August 22, 2015 slo_timeAM  Greens delay Adani until 2017  Indian energy giant Adaniâ€™s $16 billion Carmichael mine... http://t.co/n429QU4w0D"
        }
    ]
    run_assertion_tests(PTN_time, r'slo_time', time_assertion_test_cases)

    ###########################################################

    char_elongation_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
        },
        {
            "input": "a",
            "expected": "a"
        },
        {
            "input": "aa",
            "expected": "aa"
        },
        {
            "input": "aaa",
            "expected": "aaa"
        },
        {
            "input": "aaaa",
            "expected": "aaa"
        },
        {
            "input": "aaaaa",
            "expected": "aaa"
        },
        {
            "input": "abcd",
            "expected": "abcd"
        },
        {
            "input": "baaa",
            "expected": "baaa"
        },
        {
            "input": "baaaa",
            "expected": "baaa"
        },
        {
            "input": "baaab",
            "expected": "baaab"
        },
        {
            "input": "baaaab",
            "expected": "baaab"
        },
        {
            "input": "AAAAAAAAAH  (I'm still filthy about Adani but this helps somewhat) https://t.co/sN84tXZqLX",
            "expected": "AAAH  (I'm still filthy about Adani but this helps somewhat) https://t.co/sN84tXZqLX"
        }
    ]
    run_assertion_tests(PTN_elongation, r'\1\1\1', char_elongation_assertion_test_cases)

    ###########################################################

    full_tweet_assertion_test_cases = [
        {
            "input": "",
            "expected": ""
        },
        {
            "input": " ",
            "expected": " "
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
