import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import datasets
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer

def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

def main():
	# load dataset
	full_training_dataset = pd.read_csv("~/Documents/SeniorProject/data/stance/coding/auto_trainset_tok.csv")

	full_training_dataset.stance = pd.Categorical(full_training_dataset.stance, categories=["for", "against", "neutral"], ordered=True)
	full_training_dataset['code'] = full_training_dataset.stance.cat.codes

	training_data = full_training_dataset['tweet_t']
	training_labels = full_training_dataset['code']

	full_testing_dataset = pd.read_csv("~/Documents/SeniorProject/data/stance/coding/gold_20180514_majority_fixed_tok.csv")

	full_testing_dataset.stance = pd.Categorical(full_testing_dataset.stance, categories=["for", "against", "neutral"], ordered=True)
	full_testing_dataset['code'] = full_testing_dataset.stance.cat.codes

	testing_data = full_testing_dataset['tweet_t']
	testing_labels = full_testing_dataset['code']

	testing_labels = pd.Categorical(testing_labels)

	# Set the vectorizer to transform the data into inputs for classifiers
	vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char', use_idf=False)
	x_train = vectorizer.fit_transform(training_data)
	x_test = vectorizer.transform(testing_data)

	x = np.array([(x, y) for x in range(6, 10, 2) for y in range(6, 10, 2)])
	df_model = pd.DataFrame(x, columns=['layer1', 'layer2'])

	df_model['loss'] = 0
	df_model['acc'] = 0

	for index, row in df_model.iterrows():
		# create model
		model = Sequential()
		model.add(Dense(10, input_dim=12698, activation='relu'))
		model.add(Dense(row['layer1'], activation='tanh'))
		model.add(Dense(row['layer2'], activation='tanh'))
		model.add(Dense(3, activation='softmax'))

		# Compile model
		model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		model.fit(x_train, training_labels, epochs=15, batch_size=128)

		loss_and_metrics = model.evaluate(x_test, testing_labels, batch_size=128)
		print(model.metrics_names)
		print(loss_and_metrics)
		row['loss'] = loss_and_metrics[0]
		row['acc'] = loss_and_metrics[1]

	print(df_model)


if __name__ == '__main__':
    main()
