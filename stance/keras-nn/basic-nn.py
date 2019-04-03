import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
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


def create_model(optimizer='adam', row1=100, row2=50, row3=10):
    model = Sequential()

    model.add(Dense(50, input_dim=18830, activation='relu'))
    model.add(Dense(80, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(row1, activation='tanh'))
    model.add(Dense(row2, activation='tanh'))
    model.add(Dense(row3, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def main(trainset=None, testset=None, output=None):
    # load dataset
    full_training_dataset = pd.read_csv(trainset)

    full_training_dataset.stance = pd.Categorical(full_training_dataset.stance, categories=["for", "against", "neutral"], ordered=True)
    full_training_dataset['code'] = full_training_dataset.stance.cat.codes

    full_training_dataset['combined_t'] = full_training_dataset['tweet_t'] + ' ' + full_training_dataset['user_description']
    training_data = full_training_dataset['combined_t']
    training_labels = full_training_dataset['code']

    full_testing_dataset = pd.read_csv(testset)

    full_testing_dataset.stance = pd.Categorical(full_testing_dataset.stance, categories=["for", "against", "neutral"], ordered=True)
    full_testing_dataset['code'] = full_testing_dataset.stance.cat.codes

    full_testing_dataset['combined_t'] = full_testing_dataset['tweet_t'] + ' ' + full_testing_dataset['user_description']
    testing_data = full_testing_dataset['combined_t']
    testing_labels = full_testing_dataset['code']

    testing_labels = pd.Categorical(testing_labels)

    # Set the vectorizer to transform the data into inputs for classifiers
    vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char', use_idf=False)
    x_train = vectorizer.fit_transform(training_data.values.astype('U'))
    x_test = vectorizer.transform(testing_data.values.astype('U'))

    model = KerasClassifier(build_fn=create_model)

    x = np.array([(x, y, z) for x in [20, 50, 100] for y in [6, 10, 20] for z in [4, 8, 16]])
    #x = np.array([(x, y, z) for x in [20] for y in [6] for z in [4]])
    df_model = pd.DataFrame(x, columns=['layer0', 'layer1', 'layer2'])

    df_model['loss'] = 0.0
    df_model['acc'] = 0.0

    for index, row in df_model.iterrows():
        # create model
        model = create_model(row1=row['layer0'].astype(int), row2=row['layer1'].astype(int), row3=row['layer2'].astype(int))

        model.fit(x_train, training_labels, epochs=15, batch_size=128)

        loss_and_metrics = model.evaluate(x_test, testing_labels, batch_size=128)

        df_model.at[index, 'loss'] = loss_and_metrics[0]
        df_model.at[index, 'acc'] = loss_and_metrics[1]

    print(df_model)
    df_model.to_csv(output)


if __name__ == '__main__':
    Fire(main)
