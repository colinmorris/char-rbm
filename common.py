import numpy as np
import time
from short_text_codec import NonEncodableTextException

from sklearn.preprocessing import OneHotEncoder

DEBUG_TIMING = False

# Taken from StackOverflow
def timeit(f):
    if not DEBUG_TIMING:
        return f

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print 'func:%r took: %2.4f sec' % \
            (f.__name__, te - ts)
        return result

    return timed


def vectors_from_txtfile(fname, codec):
    f = open(fname)
    skipped = 0
    vecs = []
    for line in f:
        line = line.strip()
        try:
            vecs.append(codec.encode(line))
        except NonEncodableTextException:
            # Too long, or illegal characters
            skipped += 1

    print "Gathered {} vectors. Skipped {}".format(len(vecs), skipped)
    # TODO: Why default to dtype=float? Seems wasteful? Maybe it doesn't really matter. Actually, docs here seem inconsistent? Constructor docs say default float. transform docs say int.
    # TODO: should probably try using a sparse matrix here
    vecs = np.asarray(vecs)
    return OneHotEncoder(len(codec.alphabet)).fit_transform(vecs)
