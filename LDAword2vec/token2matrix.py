from stop_words import get_stop_words

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

import setting
#  import sys
import time

from doc2token import doc2array


def extract_matrix():
    """Using CountVectorizer to turn strings into word frequency matrix

    :returns: the matrix LDA gona use

    """
    time1 = time.time()

    #  Get the data from database
    texts, doc_names = doc2array()

    # Use tf (raw term count) features for LDA. Common English words, words
    # occurring in only one document or in at least 50% of the documents are
    # removed
    #  print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        max_df=0.5,
        min_df=2,
        #  max_features=n_features,
        #  preprocessor=lambda x: x,
        analyzer='word',
        stop_words='english'
        #  stop_words=get_stop_words('en')
        #  tokenizer=lda_tokenizer
    )
    tf = tf_vectorizer.fit(texts)

    if not __debug__:
        print("finished tokenizing")

    tf = tf_vectorizer.transform(texts)
    tf_feature_names = tf_vectorizer.get_feature_names()

    #  Vocabulary set
    setting.vcb = set(tf_feature_names)
    setting.stop_words = tf_vectorizer.get_stop_words()

    if not __debug__:
        print("Time for extract_matrix(): ", end='')
        print(time.time() - time1)

    return tf, tf_vectorizer, doc_names


def init_stemmer():
    """init p_stemmer

    """
    #  create tokenizer
    global tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    global en_stop
    en_stop = set(get_stop_words('en'))

    # Create p_stemmer of class PorterStemmer
    global p_stemmer
    p_stemmer = PorterStemmer()

    return


ID = 0


def lda_tokenizer(raw):
    """self defined tokenizer

    :raw: string of content in file
    :returns: array of tokens

    """
    global ID
    print("tokenizing doc: ", ID)
    ID += 1

    #  time1 = time.time()

    #  make tokens

    #  TODO: delete #
    print(raw)
    raw = raw.lower()

    global tokenizer
    tokens = tokenizer.tokenize(raw)

    global en_stop
    tokens = [i for i in tokens if i not in en_stop]

    #  print("Stop word time: ", time.time() - time1)

    for i in range(len(tokens)):
        token = tokens[i]

        if token.isdigit():
            tokens[i] = '#' * (len(token))
        else:
            tokens[i] = p_stemmer.stem(token)

    #  print("Change Ele time: ", time.time() - time1)

    # remove numbers
    #  tokens = [convert_number(i) for i in tokens]

    # remove stop words from tokens
    #  global en_stop
    #  tokens = [i for i in tokens if i not in en_stop]

    # stem tokens
    #  global p_stemmer
    #  tokens = [p_stemmer.stem(i) for i in tokens]

    return tokens


def convert_number(word):
    """convert number to special word. For example 123 to ***.

    :word: string of word
    :returns: string of word

    """
    if word.isdigit():
        word = '#'*(len(word))
    return word


init_stemmer()

if __name__ == "__main__":

    setting.init()
    tf, tf_feature_names, _ = extract_matrix()
    print("-------------------")
    print(tf)
    #  print(tf_feature_names[-1])
    #  debug
    #  print(tf.inverse_transform(texts))
