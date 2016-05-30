import argparse
import pickle
import numpy as np
import utils
import enum

import common
from short_text_codec import ShortTextCodec

class VisInit(enum.Enum):
    """Ways of initializing visible units before repeated gibbs sampling."""
    zeros = 1
    biases = 2
    uniform = 3
    spaces = 4
    train = 5

def print_samples(model, visibles):
    for v in visibles:
        print model.codec.decode(v, pretty=True)


@common.timeit
def sample_model(model, n, iters, prog, max_prob, init_method=VisInit.biases, training_examples=None):
    # Rather than starting from a random configuration of hidden or visible nodes, 
    # sample visible nodes treating their biases as softmax inputs. Intuitively,
    # this makes sense as a way to give the sampling a head start to convergence,
    # and it empirically seems to produce more natural samples after a small
    # number of iterations, compared to a uniformly random initialization over
    # visible or hidden nodes.
    vis_shape = (n, model.intercept_visible_.shape[0])
    if init_method == VisInit.biases:
        sm = np.tile(model.intercept_visible_, [n, 1]).reshape( (-1,) + model.codec.shape() )
        vis = utils.softmax_and_sample(sm).reshape(vis_shape)
    elif init_method == VisInit.zeros:
        vis = np.zeros(vis_shape)
    elif init_method == VisInit.uniform:
        vis = np.random.randint(0, 2, vis_shape)
    # This will fail if ' ' isn't in the alphabet of this model
    elif init_method == VisInit.spaces:
        vis = np.zeros( (n,) + model.codec.shape())
        vis[:,:,model.codec.char_lookup[' ']] = 1
        vis = vis.reshape(vis_shape)
    elif init_method == VisInit.train:
        assert training_examples is not None, "No training examples provided to initialize with"
        examples = common.vectors_from_txtfile(training_examples, model.codec)
        vis = examples[:n]
    else:
        raise ValueError("Unrecognized init method: {}".format(init_method))
    print_samples(model, vis)
    power = 0
    for i in range(iters):
        if prog and (i == 10**power):
            power += 1
            print "After {} iterations".format(i)
            sample = model.gibbs(vis, sample_max=max_prob)
            print_samples(model, sample)
            print
        vis = model.gibbs(vis)

    sample = model.gibbs(vis, sample_max=max_prob)
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
    parser.add_argument('--no-max', dest='nomax', action='store_true',
                        help='Default behaviour is to perform n rounds of Gibbs sampling, then one more ' +
                        'special round of sampling where we take the visible unit with the highest probability. ' +
                        'If this flag is enabled, the final round of sampling will be standard one, where we ' +
                        'sample randomly according to the softmax probabilities of visible units.')

    args = parser.parse_args()

    for model_fname in args.model_fname:
        print "Drawing samples from model defined at {}".format(model_fname)
        f = open(model_fname)
        model = pickle.load(f)
        f.close()
        sample_model(model, args.n_samples, args.iters, args.prog, max_prob=not args.nomax)
