import numpy as np
import time

from sklearn.preprocessing import OneHotEncoder

# TODO: Do a before-and-after comparison. If the non-softmax version is unambiguously worse,
# then just remove this and a lot of associated cruft. If it isn't, weep over all the numpy
# arcana you learned for nothing.
SOFTMAX = 1

# Keep it kind of small for now
NCHARS = 27
MAXLEN = 20

DEBUG_TIMING = False
def timeit(f):
    if not DEBUG_TIMING:
        return f

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print 'func:%r took: %2.4f sec' % \
          (f.__name__, te-ts)
        return result

    return timed

_CHAR2I = {' ': 26}
for (i, o) in enumerate(range(ord('a'), ord('z')+1)):
  _CHAR2I[chr(o)] = i
for (i, o) in enumerate(range(ord('A'), ord('Z')+1)):
  _CHAR2I[chr(o)] = i
  
# This happens to work out such that we get the lowercase letters as values, which is nice.
_I2CHAR = {v:k for (k, v) in _CHAR2I.iteritems()}

def decode_and_print(vec):
  """Given a one-hot vector with a bunch of floats that we interpret as softmax,
  return a corresponding string repr"""
  # Let's start by just doing max instead of sampling randomly. Easier.
  chars = []
  for position_index in range(MAXLEN):
      char_index = np.argmax(vec[position_index*NCHARS:(position_index+1)*NCHARS])
      chars.append(_I2CHAR[char_index])
  print ''.join(chars)
  
def vectorize_str(s):
  # Pad to fixed length with spaces
  return [_CHAR2I[c] for c in s] + [_CHAR2I[' '] for _ in range(MAXLEN - len(s))]
  
def vectors_from_txtfile(fname):
  f = open(fname)
  skipped = 0
  vecs = []
  for line in f:
    line = line.strip()
    if len(line) > MAXLEN:
      skipped += 1
      continue
    try:
      vecs.append(vectorize_str(line))
    except KeyError:
      # Some non-ascii chars slipped in
      skipped += 1

  print "Gathered {} vectors. Skipped {}".format(len(vecs), skipped)
  # TODO: Why default to dtype=float? Seems wasteful? Maybe it doesn't really matter. Actually, docs here seem inconsistent? Constructor docs say default float. transform docs say int. 
  # TODO: should probably try using a sparse matrix here
  vecs = np.asarray(vecs)
  print vecs.shape
  return OneHotEncoder(NCHARS).fit_transform(vecs)
