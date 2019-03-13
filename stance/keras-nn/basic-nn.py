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
from fire import Fire

from sklearn.feature_extraction.text import TfidfVectorizer

def main(trainset=None, testset=None):
	# load dataset
	full_training_dataset = pd.read_csv(trainset)

	full_training_dataset.stance = pd.Categorical(full_training_dataset.stance, categories=["for", "against", "neutral"], ordered=True)
	full_training_dataset['code'] = full_training_dataset.stance.cat.codes


#	full_training_dataset['combined_t'] = full_training_dataset['tweet_t'] + " " + full_training_dataset['user_description']
	training_data = full_training_dataset['tweet_t']
	training_labels = full_training_dataset['code']

	full_testing_dataset = pd.read_csv(testset)

	full_testing_dataset.stance = pd.Categorical(full_testing_dataset.stance, categories=["for", "against", "neutral"], ordered=True)
	full_testing_dataset['code'] = full_testing_dataset.stance.cat.codes

#	full_testing_dataset['combined_t'] = full_testing_dataset['tweet_t'] + "_" + full_testing_dataset['user_description']
	testing_data = full_testing_dataset['tweet_t']
	testing_labels = full_testing_dataset['code']

	testing_labels = pd.Categorical(testing_labels)

	# Set the vectorizer to transform the data into inputs for classifiers
	vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char', use_idf=False)
	x_train = vectorizer.fit_transform(training_data)
	x_test = vectorizer.transform(testing_data)

	x = np.array([(x, y) for x in range(2, 10, 2) for y in range(2, 10, 2)])
	df_model = pd.DataFrame(x, columns=['layer1', 'layer2'])

	df_model['loss'] = 0.0
	df_model['acc'] = 0.0

	for index, row in df_model.iterrows():
		# create model
		model = Sequential()
		model.add(Dense(10, input_dim=12698, activation='relu'))
		model.add(Dense(row['layer1'].astype(int), activation='tanh'))
		model.add(Dense(row['layer2'].astype(int), activation='tanh'))
		model.add(Dense(3, activation='softmax'))

		# Compile model
		model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		model.fit(x_train, training_labels, epochs=15, batch_size=128)

		loss_and_metrics = model.evaluate(x_test, testing_labels, batch_size=128)

		df_model.at[index, 'loss'] = loss_and_metrics[0]
		df_model.at[index, 'acc'] = loss_and_metrics[1]

	print(df_model)


if __name__ == '__main__':
    Fire(main)
