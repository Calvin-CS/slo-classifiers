import re


# Features/patterns for auto-coding
for_hashtags = ['#goadani', '#stopstopadani']
against_hashtags = ['#stopadani']
neutral_usernames = ['commsec', 'aus_business', 'financialreview', 'qanda', '4corners', '7news', '9news', 'brisbanetimes', 'ausmedia', '3novices']
for_usernames = ['adaniaustralia']

PTN_for_hashtags = re.compile('|'.join(for_hashtags))
PTN_against_hashtags = re.compile('|'.join(against_hashtags))
PTN_neutral_screennames = re.compile('|'.join(neutral_usernames))
PTN_for_screennames = re.compile('|'.join(for_usernames))

# patterns that identify individual companies
PTN_companies = [
    ('adani', re.compile(r'adani')),
    ('bhp', re.compile(r'bhp|b\.h\.p\.')),
    ('cuesta', re.compile(r'cuesta')),
    ('fortescue', re.compile(r'fortescue')),
    ('iluka', re.compile(r'iluka')),
    ('newmont', re.compile(r'newmont')),
    ('oilsearch', re.compile(r'oil.{0,3}search')),
    ('riotinto', re.compile(r'rio.{0,3}tinto')),
    ('santos', re.compile(r'santos')),
    ('whitehaven', re.compile(r'whitehaven')),
    ('woodside', re.compile(r'woodside')),
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
