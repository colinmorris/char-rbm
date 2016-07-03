import pickle
import sys

def debug_model(m):
    print m.codec.debug_description()
    print "{}th round of training".format(len(model.history['pseudo-likelihood'])+1)
    for attr in ['n_components', 'batch_size', 'n_iter']:
        print "{} = {}".format(attr, getattr(m, attr))

for fname in sys.argv[1:]:
    with open(fname) as f:
        model = pickle.load(f)
    print fname
    debug_model(model)
    print
