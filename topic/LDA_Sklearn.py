# Senior Project: LDA - Topic Modeling with Scikit Learn
# Author: Derek Fisher, dgf2
# For: CS 396, at Calvin College
# Date May 2019
# Sources: https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Number of topics
no_topics = 100

# Number of words in each topic
no_top_words = 10


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# Loads preprocessed tweets, created from the _TOK using extract_preprocessed.py.
documents = open("cleaned_tweets.txt", "r").readlines()


no_features = 1000

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=10, learning_method='online', learning_offset=50.,
                                random_state=0).fit(tf)

display_topics(lda, tf_feature_names, no_top_words)
doc_topic = lda.transform(tf)
