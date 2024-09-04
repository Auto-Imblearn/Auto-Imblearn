def init():
    """Init globle variables
    :returns: None

    """
    #  The folder name of the corpus
    global database
    #  database = "NSWSC_txt"
    #  database = "simple"
    #  database = "dataset"
    #  database = "Resamplers"
    #  database = "Classifiers"
    database = "Imputers"

    #  The folder that contains the indexes
    global folder_name
    folder_name = "index_folder"
    #  folder_name = "200index_folder"
    #  folder_name = "800index_folder"

    #  The vocabulery set for mutual exclusive
    global vcb

    #  The stop words used in LDA
    global stop_words

    #  The Word2vec model corpus size
    global train_size
    train_size = 1024*70

    #  The LDA corpus size
    global corp_size
    corp_size = 1024*800

    '''
    Index File Names {{{
    '''
    #  Trained LDA model (for change query into doc-topic distribution)
    global lda_model
    lda_model = "lda_model"

    #  Reversed Index
    global reversed_file
    reversed_file = "reversed_file"

    #  Vocabulary Vectorizer
    global vcb_file
    vcb_file = "vcb_file"

    #  Vocabulary Set
    global vcb_set
    vcb_set = "vcb_set"

    #  Doc-Topic index
    global doc_topic_index
    doc_topic_index = "doc_topic_index"

    #  document' names
    global doc_names
    doc_names = "doc_names"

    #  Word2vec Model
    global word2vec_file
    word2vec_file = "word2vec_file"
    '''
    }}}
    '''

    #  The number of topics needed to clutter
    global no_topics
    #  TODO: Change number of topics #
    #  no_topics = 70
    no_topics = 5

    #  Tops number of similar words needed when finding similar search query
    global topn
    topn = 5
