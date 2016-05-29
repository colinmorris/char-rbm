import argparse
import pickle
import numpy as np

import common
from short_text_codec import ShortTextCodec


def print_samples(model, visibles):
    for v in visibles:
        print model.codec.decode(v, pretty=True)


@common.timeit
def sample_model(model, n, iters, prog):
    # TODO: Should this actually sample according to the biases?
    rand_hidden = np.random.randint(0, 2, (n, len(model.components_)))
    vis = model._sample_visibles(rand_hidden, model.random_state)
    power = 1
    for i in range(iters):
        if prog and (i == 10**power):
            power += 1
            print "After {} iterations".format(i)
            sample = model.gibbs(vis, sample_max=True)
            print_samples(model, sample)
            print
        vis = model.gibbs(vis)

    sample = model.gibbs(vis, sample_max=True)
    print_samples(model, sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample short texts from a pickled model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_fname', metavar='model.pickle', nargs='+',
                        help='One or more pickled RBM models')
    parser.add_argument('-n', '--n-samples', dest='n_samples', type=int, default=20,
                              help='How many samples to draw')
    parser.add_argument('-i', '--iters', dest='iters', type=int, default=1000,
                              help='How many rounds of Gibbs sampling to perform before generating the outputs')
    parser.add_argument('--prog', '--progressively-sample', dest='prog', action='store_true',
                        help='Output n samples after 10 rounds of sampling, then 100, 1000... until we reach a power of 10 >=iters')

    args = parser.parse_args()

    for model_fname in args.model_fname:
        print "Drawing samples from model defined at {}".format(model_fname)
        f = open(model_fname)
        model = pickle.load(f)
        f.close()
        sample_model(model, args.n_samples, args.iters, args.prog)
