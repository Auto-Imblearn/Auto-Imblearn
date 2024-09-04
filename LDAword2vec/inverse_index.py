import setting
import time


def build_inverse(tf):
    """Build inverse index for each word

    :tf: the CountVectorizer fitted model
    :returns:
        indexex: 2d array of term, doc frequency

    """
    time1 = time.time()

    i = 0
    vcb_size = len(setting.vcb)

    if not __debug__:
        print("vcb_size:", vcb_size)

    indexes = [None] * vcb_size
    while True:
        try:
            doc = tf[i].toarray()

            for term_id in range(vcb_size):
                if doc[0][term_id] != 0:
                    if indexes[term_id] is None:
                        indexes[term_id] = []
                    indexes[term_id].append(tuple([i, doc[0][term_id]]))

            i += 1
        except IndexError:
            break

    if not __debug__:
        print("Time for build_inverse(): ", end='')
        print(time.time() - time1)

    return indexes
