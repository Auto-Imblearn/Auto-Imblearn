import os
import time
import setting
from random import sample


def doc2array():
    """Extract content from database
    returns array of string
            array of files name
    """
    time1 = time.time()

    f_num = 0
    doc_names = []
    texts = []

    cwd = os.path.join(os.getcwd(), setting.database)

    file_list = os.listdir(cwd)
    corpus_size = os.path.getsize(cwd)

    if corpus_size > setting.corp_size:
        k = len(file_list) * setting.corp_size / corpus_size
        indicies = sample(range(len(file_list)), int(k))
    else:
        indicies = range(len(file_list))

    #  for filename in os.listdir(cwd):
    for i in indicies:

        filename = file_list[i]
        #  add to file list
        doc_names.append(filename)

        f_num += 1

        f_path = os.path.join(cwd, filename)

        with open(f_path, 'r') as content_file:
            content = content_file.read()

            texts.append(content)

    if not __debug__:
        print("Time for doc2array(): ", end='')
        print(time.time() - time1)
        print(len(texts))

    return texts, doc_names
