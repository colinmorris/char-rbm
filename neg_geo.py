# TODO: Cleanup imports
import numpy as np
import time
import pickle
from sklearn.neural_network import BernoulliRBM
from sklearn.utils.fixes import expit
from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
from rbm_softmax import BernoulliRBMSoftmax
import sys

import visualization as viz
from common import *

# TODO: Remove
class BernoulliRBM_(BernoulliRBM):
    
    def gibbs_prob(self, v):
        h_ = self._sample_hiddens(v, self.random_state_)
        return self.sample_vis_probs(h_)
        
    def sample_vis_probs(self, h):
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return p



  

    
@timeit
def sample_model(model, n=100, max_iters=3):
  # TODO: Just make this a method of the RBM class?  
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
        probs = model.gibbs(sample, sample_max=True)
      else: 
        probs = model.gibbs_prob(sample)
            
      print "After {} rounds of sampling".format(10**power)
      for vec in probs:
          decode_and_print(vec)
      print
  
if __name__ == '__main__':
  if len(sys.argv) > 1:
    model_fname = sys.argv[1]
    f = open(model_fname)
    model = pickle.load(f)
    viz.visualize_hidden_activations(model, 'geo_100.txt')
    viz.receptive_fields(model)
    #sample_model(model)
  else:
    # TODO: trap ctrl+c and do sampling before bailing
    #vecs = vectors_from_txtfile('../data/geonames_us.txt')
    vecs = vectors_from_txtfile('micro_geo.txt')
    train, validation = train_test_split(vecs, test_size=0.05)
    print "X Shape : " + str(train.shape)
    if SOFTMAX:
      rbm = BernoulliRBMSoftmax( softmax_shape=(MAXLEN, NCHARS), n_components=190, learning_rate=0.05, n_iter=5, verbose=1)
    else:
      rbm = BernoulliRBM_(n_components=300, learning_rate=0.05, n_iter=50, verbose=1)
    rbm.fit(train, validation)
    f = open('model.pickle', 'wb')
    pickle.dump(rbm, f)
    f.close()
    sample_model(rbm, n=10)
