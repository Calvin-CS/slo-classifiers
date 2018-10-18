from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Single line documents from http://web.eecs.utk.edu/~berry/order/node4.html#SECTION00022000000000000000
documents = open("document.txt", "r").readlines()



#documents1 = open("dataset_slo.json", encoding = "ISO-8859-1")



# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 10


# Run LDA
lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

no_top_words = 4
no_top_documents = 4

i = 0

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        global i
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i]]))
        for i in topic.argsort():
            no_top_words -1
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])


display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)