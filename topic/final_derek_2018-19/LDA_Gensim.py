# Senior Project: LDA - Gensim Example
# Author: Derek Fisher, dgf2
# For: CS 396, at Calvin College
# Date May 2019
# Sources: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
#          https://stackoverflow.com/questions/45170093/latent-dirichlet-allocation-with-prior-topic-words

import gensim
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import random
from spacy.lang.en import English
from gensim import corpora
import pickle

# Number of topics:
NUM_TOPICS = 75

# Number of words in each topic:
num_words = 10

# The following function cleans our texts and returns a list of tokens:
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

# Filters out stop words:
en_stop = set(nltk.corpus.stopwords.words('english'))

# Defines a function to prepare the text for topic modelling:
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

text_data = []
with open("cleaned_tweets.txt") as f:
    for line in f:
        tokens = prepare_text_for_lda(line)
        if random.random() > .99:
            text_data.append(tokens)

# Creating a dictionary from the data, then convert to bag-of-words corpus
# and save the dictionary and corpus for future use.
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

# Uses LDA to find 75 topics in the data:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(NUM_TOPICS)
for topic in topics:
    print(topic)
