#  import sys
import time
import setting


def find_docs(reversed_index, que_tf, num_doc, tf_vectorizer, wordToVec):
    """Find Matched Documents

    :returns: final_docs the docs needed to be ranked

    """
    time1 = time.time()

    #  Using reversed index to find docs
    final_docs = regular_search(reversed_index, que_tf)

    # TODO  inverted index can get enough result, but may not accurate
    #  TODO Use wide range to search result
    #  if len(final_docs) >= num_doc:
    #  if not __debug__:
    #  print("Time for find_docs(): ", end='')
    #  print(time.time() - time1)
    #  return final_docs

    que_terms = que_tf.nonzero()[1]

    #  Get all docs of full permutation similar query
    #  final_docs = concept_search(
    #  final_docs,
    #  len(final_docs),
    #  tf_vectorizer.vocabulary_,
    #  que_terms,
    #  wordToVec,
    #  tf_vectorizer.get_feature_names(),
    #  reversed_index)

    while len(final_docs) < num_doc:
        if setting.topn >= 30:
            break
        setting.topn += 5
        if not __debug__:
            print("Number of similar words:", setting.topn)
        final_docs = concept_search(
            final_docs,
            len(final_docs),
            tf_vectorizer.vocabulary_,
            que_terms,
            wordToVec,
            tf_vectorizer.get_feature_names(),
            reversed_index)

    if len(final_docs) >= num_doc:
        return final_docs

    if not __debug__:
        print("Time for find_docs(): ", end='')
        print(time.time() - time1)

    return final_docs


def regular_search(reversed_index, que_tf):
    """inverse index search

    :inverse_index:
    :que_tf:
    :returns:

    """
    terms_index = list(que_tf.nonzero()[1])
    terms_index = sorted(
        terms_index, key=lambda term: len(reversed_index[term]))
    final_docs = [doc[0] for doc in reversed_index[terms_index[0]]]
    for i in range(1, len(terms_index)):
        final_docs = find_same_ele(final_docs, reversed_index[terms_index[i]])
        if final_docs == []:
            break
    #  print(final_docs)
    return set(final_docs)


def find_same_ele(list1, list2):
    """find same element in 2 list

    :list1:
    :list2:
    :returns: same elements list

    """
    final_list = []
    i = 0
    j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j][0]:
            #  final_list.append(tuple([list1[i], list1[i][1]+list2[j][1]]))
            final_list.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j][0]:
            i += 1
        elif list1[i] > list2[j][0]:
            j += 1
    return final_list


def concept_search(
        final_docs,
        doc_count,
        vocabulary,
        que_terms,
        model,
        word_name,
        reversed_index):
    """ Searching for additional docs by concept

        : final_docs     : the final output document index
        : doc_count      : the number of output document
        : vocabulary     : term-index mapping
        : que_terms      : the original query terms
        : model          : word2vec trained model
        : word_name      : vocabulary name list
        : reversed_index : reversed index for word-docs

        :returns: new final docs

    """

    #  Build (terms_index, distance) list
    query_SimTerms = [None] * len(que_terms)

    for i in range(len(que_terms)):
        if word_name[que_terms[i]] in model.wv.vocab:
            new_terms = model.wv.most_similar(
                positive=[word_name[que_terms[i]]], topn=setting.topn)

            #  print(word_name[que_terms[i]], new_terms)

            query_SimTerms[i] = [(vocabulary[term[0]], term[1])
                                 for term in new_terms]
        else:
            if not __debug__:
                print("The word rules out:", word_name[que_terms[i]])

    final_set = set([])
    final_set_flag = False
    for i in range(len(que_terms)):
        #  Union all similar term docs
        tmp_set = union_lists(
            reversed_index, query_SimTerms[i],
            que_terms[i])

        #  Intersect between different set of terms
        if not final_set_flag:
            final_set_flag = True
            final_set = tmp_set
        else:
            final_set &= tmp_set

    return final_set | final_docs


def union_lists(reversed_index, terms, query_term):
    """ Compute the union of docs for similar terms

    : reversed_index : reversed index
    : terms          : similar terms
    : term_count     : number of similar terms

    :returns: Union list

    """
    term_count = setting.topn
    #  union_set = set([])
    union_set = set([doc[0] for doc in reversed_index[query_term]])

    if terms is None:
        return union_set

    for i in range(term_count):
        union_set |= set([doc[0] for doc in reversed_index[terms[i][0]]])

    return union_set
