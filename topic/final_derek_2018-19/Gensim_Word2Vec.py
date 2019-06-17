# Senior Project: Gensim - Word2Vec model
# Author: Derek Fisher, dgf2
# For: CS 396, at Calvin College
# Date May 2019
# Sources: https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# 		   https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

import matplotlib
matplotlib.use("TkAgg")
import warnings
warnings.filterwarnings(action='ignore')
import gensim
import logging
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# dataset used to create model.
data_file = "cleaned_tweets.txt"

# Tokenizes dataset.
with open("cleaned_tweets.txt", 'rb') as f:
    for i, line in enumerate(f):
        break

def read_input(input_file):
	"""This method reads the input file which is in gzip format"""

	logging.info("reading file {0}...this may take a while".format(input_file))

	with open(input_file, 'rb') as f:
		for i, line in enumerate(f):

			if (i % 10000 == 0):
				logging.info("read {0} reviews".format(i))
			# do some pre-processing and return a list of words for each review text
			yield gensim.utils.simple_preprocess(line)


# read the tokenized reviews into a list
# each review item becomes a series of words
# so this becomes a list of lists
documents = list(read_input(data_file))
logging.info("Done reading data file")

model = gensim.models.Word2Vec(documents, size=100, window=5, min_count=100, workers=10)
model.train(documents, total_examples=len(documents), epochs=10)

# Function uses the Word2Vec model and displays it on a x-y plotted graph.
def tsne_plot(model):
	"Creates and TSNE model and plots it"
	labels = []
	tokens = []

	for word in model.wv.vocab:
		if len(word) <= 4:
			pass
		else:
			tokens.append(model[word])
			labels.append(word)

	tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=1000, random_state=23)
	new_values = tsne_model.fit_transform(tokens)

	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])

	plt.figure(figsize=(12, 12))
	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.annotate(labels[i],
					 xy=(x[i], y[i]),
					 xytext=(5, 2),
					 textcoords='offset points',
					 ha='right',
					 va='bottom')
	plt.show()

tsne_plot(model)
