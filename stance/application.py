"""This is the SLO stance classifier.

It is designed to be run as a web-service in the SLO real-time demo system and
assumes that the input files are stored in this structure:

- stance_data_root/coding/TYPE_DATERANGE_*.csv
- stance_data_root/models/TRAINCOMPANY_DATERANGE.pkl
- stance_data_root/wordvec/DATERANGE/*.vec

"""
import logging
import pickle
import os

from sklearn.metrics import f1_score

from data_utility import load_combined_data
from model_factory import ModelFactory

logger = logging.getLogger(__name__)


# TODO: Refactor this as a sub-class of Tornado.
class Application():

    def __init__(self,
                 root='.',
                 train_target='all',
                 labels=None,
                 period='',
                 model_name='svm',
                 embeddings_filename='all-100.vec',
                 rebuild_model=False,
                 ):
        """This constructor loads the SLO classification model, building
        and retraining a new one if requested.
        """
        self.target = train_target
        self.labels = labels
        self.model_name = model_name

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

        self.model_filepath = f'{root}/models/{self.target}_{period}.pkl'
        self.trainset_filepath = f'{root}/coding/auto_{period}_tok.csv'
        self.embeddings_filepath = f'{root}/wordvec/{period}/{embeddings_filename}'

        if rebuild_model or not os.path.isfile(self.model_filepath):
            self.train_save_model()

        logger.info(f'loading model from {self.model_filepath}')
        self.model = pickle.load(open(self.model_filepath, 'rb'))

    def train_save_model(self):
        """This function constructs, trains and saves an SVM model."""
        # Load the trainset.
        logger.info(f'training model using trainset: {self.trainset_filepath}')
        x_train, y_train = load_combined_data(self.trainset_filepath,
                                              self.labels,
                                              profile=True,
                                              )

        # Build the model and train it on the given data.
        model = ModelFactory.get_model(self.model_name,
                                       target=self.target,
                                       wvfp=self.embeddings_filepath,
                                       profile=True
                                       )
        model.fit(x_train, y_train)

        # Save the trained model, saving it in a file if desired.
        logger.info(f'saving model in {self.model_filepath} ')
        with open(self.model_filepath, 'wb') as model_fout:
            pickle.dump(model, model_fout)

    # TODO: Rebuild as a POST method that parses the input x_test and marshalls the result.
    def predict(self, x_test_raw):
        """This function uses the classifier to predict the stance of each
        item in the given dataset.
        TODO: Add requirements for the input file & field format.
        """
        # TODO: must normalize the tweet and profile texts
        return self.model.predict(x_test)

    def translate_predicted(self, y_predicted):
        """Converts the predicted codes to their corresponding label."""
        return [self.labels[x] for x in y_predicted]


if __name__ == "__main__":

    # root = 'c:/projects/csiro/data/stance'
    root = '/media/hdd_2/slo/stance'
    train_target = 'adani'
    labels = ['against', 'for', 'neutral', 'na']
    period = '20100101-20180510'
    model_name = 'svm'
    embeddings_filename = 'all-100.vec'

    application = Application(
        root=root,
        train_target=train_target,
        labels=labels,
        period=period,
        model_name=model_name,
        embeddings_filename=embeddings_filename,
        rebuild_model=False,
    )

    # TODO: Remove this test code; predict() will be called by the web framework.
    # Load/run the model on a given dataset for a given target company.
    train_dataset_filename = f'{root}/coding/gold_20180514_kvlinden_tok.csv'
    x_test, y_test = load_combined_data(train_dataset_filename,
                                        labels,
                                        profile=True,
                                        )
    y_predicted = application.predict(x_test)
    print(application.translate_predicted(y_predicted))
    print(f1_score(y_test, y_predicted, labels=[0, 1, 2], average='macro'))

