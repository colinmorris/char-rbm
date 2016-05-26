import numpy as np
import time
import pickle
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.fixes import expit
from sklearn.utils import check_random_state
from rbm_softmax import BernoulliRBMSoftmax
import sys

SOFTMAX = 1

# Allowed chars for geo: [a-z ] -> 26+1=27 chars

# Keep it kind of small for now
NCHARS = 27
MAXLEN = 20

_CHAR2I = {' ': 26}
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
def sample_model(model, n=100, max_iters=3):  
  #rand = np.random.randint(0, 2, (n, MAXLEN * NCHARS))
  if 1:
    # TODO: Should this actually sample according to the biases?
    rand_hidden = np.random.randint(0, 2, (n, len(model.components_)) )
    rand = model._sample_visibles(rand_hidden, model.random_state)
  else:
      rand = np.zeros( (n, MAXLEN * NCHARS) )
      # Lazy, inefficient, non-numpyistic way to do this
      for i in range(n):
        for cindex in range(MAXLEN):
          posindex = np.random.randint(0, NCHARS)
          rand[i][cindex*NCHARS + posindex] = 1
  
  sample = rand
  for power in range(max_iters):
      for _ in range(10**power):
        sample = model.gibbs(sample)
     
      if SOFTMAX:
        probs = model.gibbs_max(sample)
      else: 
        probs = model.gibbs_prob(sample)
            
      print "After {} rounds of sampling".format(10**power)
      for vec in probs:
          decode_and_print(vec)
      print
      
def receptive_fields(model):
  f = open('recep.html', 'w')
  res = '''<html><head><style>
    span {
      padding-right: 10px;
    }
    span.chars {
      display: inline-block;
      min-width: 200px;
      min-height: 1.0em;
      }
    span.neg {
      color: maroon;
      }
    span.neg, span.pos {
      display: inline-block;
      width: 300px;
      }
  </style>
  </head>
  <body>
  '''
  THRESH = 1.5
  UPPER_THRESH = THRESH*3
  def style(w):
    if w <= UPPER_THRESH:
      return "opacity: {:.2f}".format(w/UPPER_THRESH)
    return "font-size: {:.2f}em".format(w/UPPER_THRESH)
  opacity = lambda w: min(w, 1.0) / 1.0
  for component_index, h in enumerate(model.components_):
    res += '<div><h2>' + str(component_index) + '</h2>'
    for cindex in range(MAXLEN):
      weights = zip(range(NCHARS), h[cindex*NCHARS:(cindex+1)*NCHARS])
      weights.sort(key = lambda w: w[1], reverse=True)
      # Highly positive weights
      res += '<span class="pos"><span class="chars">'
      for i, w in weights:
        if w < THRESH:
          break
        char = _I2CHAR[i]
        if char == ' ':
          char = '_'
        res += '<span style="{}">'.format(style(w)) + char + '</span>'
      res += '</span>'
      maxw = weights[0][1]
      if maxw >= THRESH:
        res += '<span class="maxw">{:.1f}</span>'.format(weights[0][1])
      res += '</span>'
      
      # Highly negative weights
      res += '<span class="neg"><span class="chars">'
      for i, w in reversed(weights):
        w = -1 * w
        if w < THRESH:
          break
        char = _I2CHAR[i]
        if char == ' ':
          char = '_'
        res += '<span style="{}">'.format(style(w)) + char + '</span>'
      res += '</span>'
      minw = weights[-1][1] * -1
      if minw >= THRESH:
        res += '<span class="maxw">{:.1f}</span>'.format(minw)
      res += '</span>'
        
      res += '<br/>'
    res += '</div>'
  res += '</body></html>'
  f.write(res)
  f.close()
      
def visualize_hidden_activations(model, example_fname):
  s = '''<html><head><style>
    body {
      font-family: monospace;
      font-size: 0.9em;
      }
    .n {
      color: black;
      opacity: .2;
      }
    .y {
      color: blue;
      }
  </style></head><body><pre>'''
  vecs = vectors_from_txtfile(example_fname)
  hiddens = model._sample_hiddens(vecs, check_random_state(model.random_state))
  PADDING = 3 + 1
  s += ' '*5 + '0'
  for i in range(5*PADDING, hiddens.shape[1]*PADDING, 5*PADDING):
    s += str(i/PADDING).rjust(5*PADDING, ' ')
  s += '<br/>'
  for i, hid in enumerate(hiddens):
    #s += '{:3d}  '.format(i) + ''.join(['|' if h == 1 else '.' for h in hid]) + '<br/>'
    s += '{:3d}  '.format(i) + ''.join(
      ['<span class="{}">|{}</span>'.format("y" if h == 1 else "n", " "*(PADDING-1)) for h in hid]
      )
    s += ' ' + str(sum(hid))
    s += '<br/>'
  
  
  s += ' ' * 5 + ''.join([str(sum(active)).ljust(PADDING, ' ') 
    for active in hiddens.T])
  s += '</pre></body></html>'
  fout = open('hidden_activations.html', 'w')
  fout.write(s)
  
if __name__ == '__main__':
  if len(sys.argv) > 1:
    model_fname = sys.argv[1]
    f = open(model_fname)
    model = pickle.load(f)
    visualize_hidden_activations(model, 'geo_100.txt')
    receptive_fields(model)
    #sample_model(model)
  else:
    # TODO: trap ctrl+c and do sampling before bailing
    #vecs = vectors_from_txtfile('../data/geonames_us.txt')
    vecs = vectors_from_txtfile('mini_geo.txt')
    print "X Shape : " + str(vecs.shape)
    if SOFTMAX:
      rbm = BernoulliRBMSoftmax( softmax_shape=(MAXLEN, NCHARS), n_components=190, learning_rate=0.05, n_iter=4, verbose=1)
    else:
      rbm = BernoulliRBM_(n_components=300, learning_rate=0.05, n_iter=50, verbose=1)
    fit(rbm, vecs)
    f = open('model.pickle', 'wb')
    pickle.dump(rbm, f)
    f.close()
    sample_model(rbm)
