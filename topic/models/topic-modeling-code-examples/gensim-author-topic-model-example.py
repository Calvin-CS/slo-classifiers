"""
Example from Gensim API documentation on the ATM Class.
"""

########################################################################################

from gensim.models import AuthorTopicModel
from gensim.corpora import mmcorpus
from gensim.test.utils import common_dictionary, datapath, temporary_file


########################################################################################

def example_1():
    """
    Example code from Gensim documentation on author-topic class.
    :return:
    """
    author2doc = {
        'john': [0, 1, 2, 3, 4, 5, 6],
        'jane': [2, 3, 4, 5, 6, 7, 8],
        'jack': [0, 2, 4, 6, 8]
    }

    corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

    print("Corpus contents:")
    print(f"{corpus}\n")

    for document in corpus:
        print(f"{document}")

    print("Dictionary contents:")
    print(f"{common_dictionary}\n")
    print(f"{common_dictionary.token2id}")

    with temporary_file("serialized") as s_path:
        model = AuthorTopicModel(
            corpus, author2doc=author2doc, id2word=common_dictionary, num_topics=4,
            serialized=True, serialization_path=s_path
        )

        model.update(corpus, author2doc)  # update the author-topic model with additional documents

    # construct vectors for authors
    author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]
    print(author_vecs)


########################################################################################

# Call the function.
example_1()
