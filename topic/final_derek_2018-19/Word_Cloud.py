# Senior Project: Word Cloud generator
# Author: Derek Fisher, dgf2
# For: CS 396, at Calvin College
# Date May 2019
# Sources: https://towardsdatascience.com/identify-top-topics-using-word-cloud-9c54bc84d911
#          https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Load the desired dataset.
dataset = pd.read_csv("../Data/dataset_20100101-20180510_tok.csv")
# dataset = pd.read_csv("../Data/gold_20180514_majority_fixed_tok.csv")
# dataset = pd.read_csv("../Data/gold_against_20180514_majority_fixed_tok.csv")
# dataset = pd.read_csv("../Data/gold_for_20180514_majority_fixed_tok.csv")
# dataset = pd.read_csv("../Data/auto_10688_20100101-20180706_1.csv")

# Coverts all stings to lowercase.
all_text = ' '.join(dataset['text'].str.lower())

# List of stop words to be removed
stopwords = ["slo_url", "slo_mention", "word_n", "slo_year", "slo_cash", "woodside", "auspol", "adani", "stopadani",
               "ausbiz", "santos", "whitehaven", "tinto", "fortescue", "bhp", "adelaide", "billiton", "csg", "nswpol",
               "nsw", "lng", "don", "rio", "pilliga", "australia", "asx", "just", "today", "great", "says", "like",
               "big", "better", "rite", "would", "SCREEN_NAME", "mining", "former", "qldpod", "qldpol", "qld", "wr",
               "melbourne", "andrew", "fuck", "spadani", "greg", "th", "australians", "http", "https", "rt", "goadani",
             "co", "amp", "riotinto", "carmichael", "abbot", "bill shorten", "via", "smh", "santosltd", "said",
             "annastaciamp", "qldlabor", "without", "bhp", "4corners", "christinemilne", "cfmeu", "jimblebar",
             "bhpbilliton", "oz", "two", "appea", "media_release", "ly", "will", "fmg", "7news", "sto", "3novices"] + list(STOPWORDS)


# Creates a word cloud from dataset and displays the model using matplot.
wordcloud = WordCloud(stopwords=stopwords, background_color="black", max_words=999, max_font_size=40).generate(all_text)
plt.imshow(wordcloud, interpolation='spline16')
plt.axis("off")
plt.show()
