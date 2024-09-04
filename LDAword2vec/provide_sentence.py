import os
from random import sample

import nltk.data

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

import setting

#  from token2matrix import convert_number


class GetSentence(object):
    """Get one sentence each time from a provided dir"""

    def __init__(self, dirName):
        self.dirName = dirName

        #  create tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # create English stop words list
        self.en_stop = setting.stop_words

        # Create p_stemmer of class PorterStemmer
        self.p_stemmer = PorterStemmer()

    def __iter__(self):
        """provide one sentence at each iteration
        :returns:

        """
        #  yield os.listdir(self.dirName)

        file_list = os.listdir(self.dirName)
        corpus_size = os.path.getsize(self.dirName)

        if corpus_size > setting.train_size:
            k = len(file_list) * setting.train_size / corpus_size
            indicies = sample(range(len(file_list)), int(k))
        else:
            indicies = range(len(file_list))
        #  for fName in os.listdir(self.dirName):
        for i in indicies:
            fName = file_list[i]
            file_path = os.path.join(self.dirName, fName)
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            fp = open(file_path)
            with open(file_path) as fp:
                content = fp.read()
                for sentence in tokenizer.tokenize(content):
                    yield self.word_tokennizer(sentence)

    def word_tokennizer(self, sentence):
        """Pre-processing of sentences to match LDA

        :sentence: file sentences
        :returns: stemmed sentences

        """
        #  make tokens
        tokens = self.tokenizer.tokenize(sentence)

        # remove numbers
        #  tokens = [convert_number(i) for i in tokens]

        # remove stop words from tokens
        tokens = [i for i in tokens if i not in self.en_stop]

        # stem tokens
        #  tokens = [self.p_stemmer.stem(i) for i in tokens]

        # Delete unuseful word
        tokens = [i for i in tokens if i in setting.vcb]

        return tokens


if __name__ == "__main__":
    hi = GetSentence("test")

    for i in hi:
        print("------------")
        print(i)
        print("------------")
