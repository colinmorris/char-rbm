import numpy as np
import time
import pickle
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.fixes import expit
from rbm_softmax import BernoulliRBMSoftmax

SOFTMAX = 1

# Allowed chars for actors: [a-z,.'- ] -> 26+5=31 chars

# Keep it kind of small for now
NCHARS = 31
MAXLEN = 10

_CHAR2I = {',': 26, '.': 27, "'": 28, ' ': 29, '-': 30}
for (i, o) in enumerate(range(ord('a'), ord('z')+1)):
  _CHAR2I[chr(o)] = i
for (i, o) in enumerate(range(ord('A'), ord('Z')+1)):
  _CHAR2I[chr(o)] = i
  
# This happens to work out such that we get the lowercase letters as values, which is nice.
_I2CHAR = {v:k for (k, v) in _CHAR2I.iteritems()}
  
class BernoulliRBM_(BernoulliRBM):
    
    def gibbs_prob(self, v):
        h_ = self._sample_hiddens(v, self.random_state_)
        return self.sample_vis_probs(h_)
        
    def sample_vis_probs(self, h):
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return p

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print 'func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts)
        return result

    return timed

def vectorize_str(s):
  # Pad to fixed length with spaces
  return [_CHAR2I[c] for c in s] + [_CHAR2I[' '] for _ in range(MAXLEN - len(s))]

@timeit
def vectors_from_txtfile(fname):
  f = open(fname)
  skipped = 0
  vecs = []
  for line in f:
    line = line.strip()
    if len(line) > MAXLEN:
      skipped += 1
      continue
    vecs.append(vectorize_str(line))

  print "Gathered {} vectors. Skipped {}".format(len(vecs), skipped)
  # TODO: Why default to dtype=float? Seems wasteful? Maybe it doesn't really matter. Actually, docs here seem inconsistent? Constructor docs say default float. transform docs say int. 
  # TODO: should probably try using a sparse matrix here
  vecs = np.asarray(vecs)
  print vecs.shape
  return OneHotEncoder(NCHARS).fit_transform(vecs)
  
@timeit
def fit(model, X):
    model.fit(X)
    
def decode_and_print(vec):
  """Given a one-hot vector with a bunch of floats that we interpret as softmax,
  return a corresponding string repr"""
  # Let's start by just doing max instead of sampling randomly. Easier.
  chars = []
  for position_index in range(MAXLEN):
      char_index = np.argmax(vec[position_index*NCHARS:(position_index+1)*NCHARS])
      chars.append(_I2CHAR[char_index])
  print ''.join(chars)

    
@timeit
def sample(model, n=100, max_iters=20):  
  #rand = np.random.randint(0, 2, (n, MAXLEN * NCHARS))
  rand = np.zeros( (n, MAXLEN * NCHARS) )
  # Lazy, inefficient, non-numpyistic way to do this
  for i in range(n):
    for cindex in range(MAXLEN):
      posindex = np.random.randint(0, NCHARS)
      rand[i][cindex*NCHARS + posindex] = 1
  
  sample = rand
  for i in range(max_iters):
      sample = model.gibbs(sample)
     
  if SOFTMAX:
    probs = sample
    print probs[0]
  else: 
    probs = model.gibbs_prob(sample)
        
  for vec in probs:
      decode_and_print(vec)
  
if __name__ == '__main__' and 0:
  # TODO: trap ctrl+c and do sampling before bailing
  vecs = vectors_from_txtfile('actors_cleaned_rand.txt')
  print "X Shape : " + str(vecs.shape)
  if SOFTMAX:
    rbm = BernoulliRBMSoftmax( softmax_shape=(MAXLEN, NCHARS), n_components=250, learning_rate=0.06, n_iter=20, verbose=1)
  else:
    rbm = BernoulliRBM_(n_components=300, learning_rate=0.05, n_iter=50, verbose=1)
  fit(rbm, vecs)
  sample(rbm)
  f = open('model.pickle', 'wb')
  pickle.dump(rbm, f)
  f.close()
