import sys
import pickle

for model_fname in sys.argv[1:]:
    with open(model_fname) as f:
        model = pickle.load(f)
        print model_fname
        if not hasattr(model, 'history'):
            print "WARNING: skipping old model without history"
            continue
        for (roundno, l9s) in enumerate(model.history['pseudo-likelihood']):
            example = l9s[0]
            if isinstance(example, tuple):
                print "WARNING: skipping dumb old-style model with dumb tuples of energy means"
                continue
            elif not isinstance(example, float):
                print "WARNING: got this thing: {}".format(type(example))
                continue
            print "training round #{}".format(roundno+1)
            print "\t".join("{:.2f}".format(l9) for l9 in l9s)
        print
