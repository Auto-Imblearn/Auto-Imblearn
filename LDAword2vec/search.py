import os
import sys
import time

import setting
import process_index

from doc_search import find_docs
from sorting_docs import sorting_docs


def search():
    """Search for query using LDA and Word2Vec
    :returns: Doc names

    """

    #  Get user query request
    query, num_doc = get_input()

    #  Init global variables
    setting.init()

    #  Get index needed to do search
    lda, doc_names, tf_vectorizer, reversed_index,\
        doc_topic_index, wordToVec = get_index()

    #  Pre-process query terms
    que_tf = tf_vectorizer.transform([query])

    #  If no result, then print nothing
    if que_tf.getnnz == 0:
        print("There is no search result for the query")
        sys.exit()

    time1 = time.time()
    #  Find Matched Documents ID
    final_docs = find_docs(
        reversed_index,
        que_tf,
        num_doc,
        tf_vectorizer,
        wordToVec)
    #  final_docs = {0,1,2,3,4,5}

    #  Ranking docs
    que_dis = lda.transform(que_tf)[0]


    #  Sorting final docs using LDA
    final_docs = sorting_docs(final_docs, doc_topic_index, que_dis)

    if not __debug__:
        print("Time for searching is: ", end="")
        print(time.time() - time1)
    #  Print document names of final result
    print_docNames(final_docs, doc_names, num_doc)

    return


def get_input():
    """get query and number of docs needed
    :returns: query

    """
    query = sys.argv[1]
    #  query = input("Type the query: ")

    #  Find the number of docs needed
    while True:
        try:
            #  num_doc = input("Type how many docs do you need: ")
            num_doc = sys.argv[2]
            num_doc = int(num_doc)
            break
        except Exception:
            print("Please only type integer!")

    return query, num_doc


def print_docNames(final_docs, doc_names, num_doc):
    """print document names of search result

    if results have enough documents, print desired results, else print what
    we have

    :final_docs:
    :doc_names:
    :returns: None

    """
    print_count = len(final_docs)
    #  for i in range(len(final_docs)):
    if len(final_docs) >= num_doc:
        print_count = num_doc

    for i in range(print_count):
        index = final_docs[i][0]
        #  print(doc_names[index])
        print(doc_names[index], final_docs[i][1])


def get_index():
    """get index
    :returns:

    """
    time1 = time.time()
    if not os.path.isdir(setting.folder_name):
        if not __debug__:
            print("Building index")
        os.makedirs(setting.folder_name)
        return process_index.build_index()
    else:
        if not __debug__:
            print("Fetching index")
        return process_index.fetch_index()

    if not __debug__:
        print("Time for get_index(): ", end='')
        print(time.time()-time1)


if __name__ == "__main__":
    search()
