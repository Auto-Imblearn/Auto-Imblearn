from sklearn.decomposition import LatentDirichletAllocation
import time
#  from sklearn.feature_extraction.text import CountVectorizer

from token2matrix import extract_matrix
import setting


def fit_model():
    """Fit LDA

    :returns:

    """
    time1 = time.time()

    #  Pre-processing
    tf, tf_vectorizer, doc_names = extract_matrix()

    #  Fit the modle
    lda = LatentDirichletAllocation(
        n_topics=setting.no_topics,
        max_iter=5,
        learning_method='online',
        learning_offset=50.,
        random_state=0).fit(tf)

    if not __debug__:
        print("Time for fit_model(): ", end='')
        print(time.time() - time1)

    #  print("Finish LDA training")

    return lda, tf, tf_vectorizer, doc_names


def display_docs(model, doc_names, no_top_words, tf):
    """
    For debug
    Deisplay the doc topics distrubution

    """
    for doc_idx, topic in enumerate(model.transform(tf)):
        print(doc_idx, doc_names[doc_idx])
        print(" ".join(["topic"+str(i)
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

    print()


def display_topic(model, feature_names, no_top_words):
    """
    For debug
    Deisplay the topic words distrubution

    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


if __name__ == "__main__":
    #  for debug

    flag = 0
    if flag == 1:
        no_top_words = 10
        lda, tf, tf_vectorizer, _ = fit_model()
        display_topic(lda, tf_vectorizer.tf_feature_names(), no_top_words)
    else:
        no_top_topics = 3
        lda, tf, _, doc_names = fit_model()
        display_docs(lda, doc_names, no_top_topics, tf)
