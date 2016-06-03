import numpy as np
import time
from collections import Counter

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




def vectors_from_txtfile(fname, codec, limit=-1, mutagen=None):
    f = open(fname)
    skipped = Counter()
    vecs = []
    for line in f:
        line = line.strip()
        try:
            vecs.append(codec.encode(line, mutagen=mutagen))
            if len(vecs) == limit:
                break
        except NonEncodableTextException as e:
            # Too long, or illegal characters
            skipped[e.reason] += 1

    # TODO: logging.debug
    print "Gathered {} vectors. Skipped {} ({})".format(len(vecs), 
        sum(skipped.values()), dict(skipped))
    # TODO: Why default to dtype=float? Seems wasteful? Maybe it doesn't really matter. Actually, docs here seem inconsistent? Constructor docs say default float. transform docs say int.
    # TODO: should probably try using a sparse matrix here
    vecs = np.asarray(vecs)
    return OneHotEncoder(len(codec.alphabet)).fit_transform(vecs)
