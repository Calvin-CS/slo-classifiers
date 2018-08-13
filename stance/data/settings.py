import re

# Features/patterns for auto-coding neutral tweets
neutral_usernames = ['commsec', 'aus_business', 'financialreview', '4corners', '7news', '9news',
                     'brisbanetimes', '3novices', 'smh', 'sbs', 'theheraldsun', 'australian', 'couriermail',
                     'abcnews', 'skynewsaust', 'qanda']

PTN_neutral_screennames = re.compile('|'.join(neutral_usernames))

PTN_company_usernames = re.compile('|'.join(['adaniaustralia', 'bhp', 'santosltd', 'fortescuenews', 'riotinto']))

# List of companies to put in the training set.
company_list = [
    'adani',
    'bhp',
    'santos',
    'riotinto',
    'fortescue',
]

# Regular expression pattern dictionary for each company of all stance-for search patterns
# NOTE: #gobhp, #gosantos, #goriotinto, #gofortescue, have no results as of 7/28/18
PTN_for = {
    'adani' : re.compile('|'.join(['#goadani', '#stopstopadani'])),
    'bhp' : re.compile('|'.join(['#gobhp', 'inspir', 'potential', 'innovat', 'women', 'woman', 'gender', 'leadership', 'apprentice', 'productivity', 'health', 'efficien'])),
    'santos' : re.compile('|'.join(['#gosantos', 'inspir', 'potential', 'innovat', 'women', 'woman', 'gender', 'leadership', 'apprentice', 'productivity', 'health', 'efficien'])),
    'riotinto' : re.compile('|'.join(['#goriotinto', 'inspir', 'potential', 'innovat', 'women', 'woman', 'gender', 'leadership', 'apprentice', 'productivity', 'health', 'efficien'])),
    'fortescue' : re.compile('|'.join(['#gofortescue', 'inspir', 'potential', 'innovat', 'women', 'woman', 'gender', 'leadership', 'apprentice', 'productivity', 'health', 'efficien']))
}

# Regular expression pattern dictionary for each company of all stance-against search patterns
# NOTE: #stopriotinto, #stopfortescue, have no results as of 7/28/18
PTN_against = {
    'adani' : re.compile('|'.join(['#stopadani'])),
    'bhp' : re.compile('|'.join(['#stopbhp', 'protest', 'stopbhp', 'csg', 'nocoal', 'nonewcoal', 'climatechange', 'shenhua', 'caroona', 'risk'])),
    'santos' : re.compile('|'.join(['#stopsantos', 'protest', 'csg', 'nocoal', 'nonewcoal', 'climatechange', 'risk'])),
    'riotinto' : re.compile('|'.join(['#stopriotinto', 'protest', 'csg', 'nocoal', 'nonewcoal', 'climatechange', 'risk'])),
    'fortescue' : re.compile('|'.join(['#stopfortescue', 'protest', 'csg', 'nocoal', 'nonewcoal', 'climatechange', 'risk']))
}


# patterns that identify individual companies
PTN_companies = [
    ('adani', re.compile(r'adani'), 'adaniaustralia'),
    ('bhp', re.compile(r'bhp|b\.h\.p\.'), 'bhp'),
    ('cuesta', re.compile(r'cuesta'), 'cuestacoal'),
    ('fortescue', re.compile(r'fortescue'), 'fortescuenews'),
    ('iluka', re.compile(r'iluka'), 'ilukaresources'),
    ('newmont', re.compile(r'newmont'), 'newmontau'),
    ('oilsearch', re.compile(r'oil.{0,3}search'), 'oilsearchltd'),
    ('riotinto', re.compile(r'rio.{0,3}tinto'), 'riotinto'),
    ('santos', re.compile(r'santos'), 'santosltd'),
    ('whitehaven', re.compile(r'whitehaven'), 'whitehavencoal'),
    ('woodside', re.compile(r'woodside'), 'woodsideenergy'),
]

# Patterns for important tweet sub-strings
PTN_rt = re.compile(r'^(RT @\w+: )')
PTN_url = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
PTN_concatenated_url = re.compile(r'(.)http')
PTN_mention = re.compile(r'@[a-zA-Z_0-9]+')
PTN_stock_symbol = re.compile(r'$[a-zA-Z]+')
PTN_hash = re.compile(r'#\w+')
PTN_whitespace = re.compile(r'\s')
PTN_elongation = re.compile(r'(.)\1{2,}')
PTN_year = re.compile(r'[12][0-9]{3}')
PTN_time = re.compile(r'[012]?[0-9]:[0-5][0-9]')
# https://stackoverflow.com/a/13848829
PTN_cash = re.compile(r'\$(?=\(.*\)|[^()]*$)\(?\d{1,3}(,?\d{3})?(\.\d\d?)?\)?([bmk]| hundred| thousand| million| billion)?')
