# Usage: You can run this by passing in all the necessary parameters, however, it is recommended that you
# explore the /storage/sloclassifiers/grid_search_25x_run.sh file and run it using something like that.
# If you want to run this stand-alone, then you can run it as follows:
# python3 basic-nn.py path/to/traindata.csv path/to/testdata.csv desired/path/to/result_file.csv

from fire import Fire
import itertools
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


def create_model(optimizer='adam', network_structure=[10, 10], input_dim=18829):
    model = Sequential()

    model.add(Dense(network_structure[0], input_dim=input_dim, activation='relu'))
    for i in range(1, len(network_structure)):
        model.add(Dense(network_structure[i], activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def main(trainset=None, testset=None, output=None):
    # load training dataset
    full_training_dataset = pd.read_csv(trainset)

    # Create coding for target labels
    full_training_dataset.stance = pd.Categorical(full_training_dataset.stance, categories=["for", "against", "neutral"], ordered=True)
    full_training_dataset['code'] = full_training_dataset.stance.cat.codes

    # Create a synthetic feature which combines tweet text and user description text
    full_training_dataset['combined_t'] = full_training_dataset['tweet_t'] + ' ' + full_training_dataset['user_description']

    # Grab the relevant feature and target labels
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

    possibilities = [10, 25, 50, 100, 500, 1000]
    num_rows = [1, 2, 3, 4]

    df_model = pd.DataFrame(columns=['network_structure', 'loss', 'acc', 'f1'])

    index = 0

    for n in range(0, len(num_rows)):
        for i in list(itertools.permutations(possibilities, num_rows[n])):
            model = create_model(network_structure=i, input_dim=x_train.shape[1])

            model.fit(x_train, training_labels, epochs=5, batch_size=128)

            loss_and_metrics = model.evaluate(x_test, testing_labels, batch_size=128)
            prediction_y = model.predict(x_test, batch_size=128)
            y_pred = [None] * len(prediction_y)
            for j in range(0, len(prediction_y)):
                y_pred[j] = np.argmax(prediction_y[j])

            df_model.loc[index] = [" ".join(str(x) for x in i),
                             loss_and_metrics[0],
                             loss_and_metrics[1],
                             f1_score(testing_labels, y_pred, average='macro')]
            index += 1

    print(df_model)
    df_model.to_csv(output)


if __name__ == '__main__':
    Fire(main)
