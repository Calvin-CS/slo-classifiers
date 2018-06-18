# Sentiment Classifier

The sentiment classifier is a legacy system; it was replaced by a stance
classifier in early 2018.

Per an email from Brian Jin, Dec 5, there were (at least) four sentiment classifiers associated with SLO.
- *LASC-Sentiment-Classifier - Mac Kim - SGDClassifier trained on 15K general tweets (SemEval 2013?) - This was the classifier
used by the real-time system in early 2018.
- Visie-Twitter-Sentiment - Louis Tiao - I can't find this one, but it appears to be older and to use a MaxEntropy classifier.
- CNN Classifier - Dai - Trained on the IMDB tagged sentiment dataset (http://ai.stanford.edu/~amaas/data/sentiment/) using Keras/Tensorflow
- Stanford SentimentAnnotator - Stanford - Trained on IMDB as well (https://stanfordnlp.github.io/CoreNLP/sentiment.html)
