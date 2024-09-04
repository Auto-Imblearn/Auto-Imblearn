import os
import pickle
import time

import setting

from wordToVec import train_word2vec
from matrix2LDA import fit_model
from inverse_index import build_inverse

#  global lda model
index_flag = False
fitted_lda = None


def build_index():
    """ Fit the model and build the index

    :returns: index

    """
    time1 = time.time()

    #  Train lda model
    lda, tf, tf_vectorizer, doc_names = fit_model()

    #  Get vocabulary set
    setting.vcb = set(tf_vectorizer.get_feature_names())

    #  Build inverse index
    reversed_index = build_inverse(tf)

    # Train word2vec model
    wordToVec = train_word2vec()

    #  Save trained LDA model
    lda_model_index = os.path.join(setting.folder_name, setting.lda_model)
    with open(lda_model_index, "wb") as index_file:

        #  save lda model as index file
        pickle.dump(lda, index_file)

    #  Save Reversed Index
    reversed_path = os.path.join(setting.folder_name, setting.reversed_file)
    with open(reversed_path, "wb") as index_file:
        pickle.dump(reversed_index, index_file)

    #  Save Doc-Topic index
    doc_topic_path = os.path.join(setting.folder_name, setting.doc_topic_index)
    with open(doc_topic_path, "wb") as index_file:
        doc_topic_index = lda.transform(tf)
        pickle.dump(doc_topic_index, index_file)

    #  Save vocabulary vectorizer
    vcb_path = os.path.join(setting.folder_name, setting.vcb_file)
    with open(vcb_path, "wb") as index_file:
        pickle.dump(tf_vectorizer, index_file)

    #  Save vocabulary set
    vcb_set_path = os.path.join(setting.folder_name, setting.vcb_set)
    with open(vcb_set_path, "wb") as index_file:
        pickle.dump(setting.vcb, index_file)

    #  Save document' names
    doc_path = os.path.join(setting.folder_name, setting.doc_names)
    with open(doc_path, "wb") as index_file:
        pickle.dump(doc_names, index_file)

    #  Save Doc-Topic index
    word2vec_path = os.path.join(setting.folder_name, setting.word2vec_file)
    with open(word2vec_path, "wb") as index_file:
        pickle.dump(wordToVec, index_file)

    #  Now we have the index files we need
    global index_flag
    index_flag = True

    if not __debug__:
        print("Time for build_index(): ", end='')
        print(time.time() - time1)

    return lda, doc_names, tf_vectorizer, reversed_index, doc_topic_index, wordToVec


def fetch_index():
    """fetch already built index
    :returns: index

    """
    time1 = time.time()

    global fitted_lda
    global index_flag
    if index_flag:
        #  already have the data
        lda = fitted_lda
    else:
        #  Fetch Doc-Topic index
        doc_topic_path = os.path.join(
            setting.folder_name, setting.doc_topic_index)
        with open(doc_topic_path, "rb") as index_file:
            doc_topic_index = pickle.load(index_file)

        #  Fetch trained lda model
        lda_model_index = os.path.join(setting.folder_name, setting.lda_model)
        with open(lda_model_index, "rb") as index_file:
            unpickler = pickle.Unpickler(index_file)
            lda = unpickler.load()
        #  fitted_lda = lda

        #  Fetch Reversed Index
        reversed_path = os.path.join(setting.folder_name, setting.reversed_file)
        with open(reversed_path, "rb") as index_file:
            reversed_index = pickle.load(index_file)

        #  Fetch vocabulary vectorizer
        vcb_path = os.path.join(setting.folder_name, setting.vcb_file)
        with open(vcb_path, "rb") as index_file:
            tf_vectorizer = pickle.load(index_file)

        #  Fetch vocabulary set
        vcb_set_path = os.path.join(setting.folder_name, setting.vcb_set)
        with open(vcb_set_path, "rb") as index_file:
            setting.vcb = pickle.load(index_file)

        #  Fetch document' names
        doc_path = os.path.join(setting.folder_name, setting.doc_names)
        with open(doc_path, "rb") as index_file:
            doc_names = pickle.load(index_file)

        word2vec_path = os.path.join(setting.folder_name, setting.word2vec_file)
        with open(word2vec_path, "rb") as index_file:
            wordToVec = pickle.load(index_file)

    if not __debug__:
        print("Time for fetch_index(): ", end='')
        print(time.time() - time1)

    return lda, doc_names, tf_vectorizer, reversed_index, doc_topic_index, wordToVec
