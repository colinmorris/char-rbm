from __future__ import division
import argparse
import pickle
import numpy as np
import utils
import enum

import common
from short_text_codec import ShortTextCodec

MAX_PROG_SAMPLE_INTERVAL = 10000

LINEAR_ANNEAL = 1

class VisInit(enum.Enum):
    """Ways of initializing visible units before repeated gibbs sampling."""
    zeros = 1
    # Treat visible biases as softmax
    biases = 2
    # Turn on each unit (not just each one-hot vector) with p=.5
    uniform = 3
    spaces = 4
    padding = 7 # Old models use ' ' as filler, making this identical to the above
    # Training examples
    train = 5
    # Choose a random length. Fill in that many uniformly random chars. Fill the rest with padding character.
    chunks = 6
    # Use training examples but randomly mutate non-space/padding characters. Only the "shape" is preserved.
    silhouettes = 8
    # Valid one-hot vectors, each chosen uniformly at random
    uniform_chars = 9

class BadInitMethodException(Exception):
    pass

def starting_visible_configs(init_method, n, model, training_examples_fname=None):
    """Return an ndarray of n visible configurations for the given model
    according to the specified init method (which should be a member of the VisInit enum)
    """
    vis_shape = (n, model.intercept_visible_.shape[0])
    maxlen, nchars = model.codec.maxlen, model.codec.nchars
    if init_method == VisInit.biases:
        sm = np.tile(model.intercept_visible_, [n, 1]).reshape( (-1,) + model.codec.shape() )
        return utils.softmax_and_sample(sm).reshape(vis_shape)
    elif init_method == VisInit.zeros:
        return np.zeros(vis_shape)
    elif init_method == VisInit.uniform:
        return np.random.randint(0, 2, vis_shape)
    # This will fail if ' ' isn't in the alphabet of this model
    elif init_method == VisInit.spaces or init_method == VisInit.padding:
        fillchar = {VisInit.spaces: ' ', VisInit.padding: model.codec.filler}[init_method]
        vis = np.zeros( (n,) + model.codec.shape())
        try:
            fill = model.codec.char_lookup[fillchar]
        except KeyError:
            raise BadInitMethodException(fillchar + " is not in model alphabet")

        vis[:,:,fill] = 1
        return vis.reshape(vis_shape)
    elif init_method == VisInit.train or init_method == VisInit.silhouettes:
        assert training_examples_fname is not None, "No training examples provided to initialize with"
        mutagen = model.codec.mutagen_silhouettes if init_method == VisInit.silhouettes else None
        examples = common.vectors_from_txtfile(training_examples_fname, model.codec, limit=n, mutagen=mutagen)
        return examples
    elif init_method == VisInit.chunks or init_method == VisInit.uniform_chars:
        # This works, but probably isn't idiomatic numpy.
        # I don't think I'll ever write idiomatic numpy.

        # Start w uniform dist
        char_indices = np.random.randint(0, nchars, (n,maxlen))
        if init_method == VisInit.chunks:
            # Choose some random lengths
            lengths = np.clip(maxlen*.25 * np.random.randn(n) + (maxlen*.66), 1, maxlen
                ).astype('int8').reshape(n, 1)
            _, i = np.indices((n, maxlen))
            char_indices[i>=lengths] = model.codec.char_lookup[model.codec.filler]
        
        # TODO: This is a useful little trick. Make it a helper function and reuse it elsewhere?
        return np.eye(nchars)[char_indices.ravel()].reshape(vis_shape)
    else:
        raise ValueError("Unrecognized init method: {}".format(init_method))


def print_sample_callback(sample_strings, i, energy=None):
    print "\t----i={}----".format(i)
    if energy is not None:
        print "\n".join('{}\t{:.4f}'.format(t[0], t[1]) for t in zip(sample_strings, energy))
    else:
        print "\n".join(sample_strings)

@common.timeit
def sample_model(model, n, iters, sample_iter_indices, 
                 start_temp=1.0, final_temp=1.0,
                 callback=print_sample_callback, init_method=VisInit.biases, training_examples=None, sample_energy=False):

    vis = starting_visible_configs(init_method, n, model, training_examples)
    temp = start_temp
    temp_decay = (final_temp/start_temp)**(1/iters)
    temp_delta = (final_temp-start_temp)/iters
    next_sample_metaindex = 0
    for i in range(iters):
        if i == sample_iter_indices[next_sample_metaindex]:
            # Time to take samples
            sample_strings = [model.codec.decode(v, pretty=True, strict=False) for v in vis]
            if sample_energy:
                energy = model._free_energy(vis)
                callback(sample_strings, i, energy)
            else:
                callback(sample_strings, i)
            next_sample_metaindex += 1
            if next_sample_metaindex == len(sample_iter_indices):
                break
        vis = model.gibbs(vis, temp)
        # XXX: hacks, experimenting
        if LINEAR_ANNEAL:
            temp += temp_delta
        elif 0:
            temp *= temp_decay
        elif i == (iters*7)//10:
            temp = temp / 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample short texts from a pickled model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_fname', metavar='model.pickle', nargs='+',
                        help='One or more pickled RBM models')
    parser.add_argument('-n', '--n-samples', dest='n_samples', type=int, default=30,
                              help='How many samples to draw')
    parser.add_argument('-i', '--iters', dest='iters', type=int, default=10**4,
                              help='How many rounds of Gibbs sampling to perform before generating the outputs')
    parser.add_argument('--prog', '--progressively-sample', dest='prog', action='store_true',
                        help='Output n samples after 0 rounds of sampling, then 1, 10, 100, 1000... until we reach a power of 10 >=iters')
    parser.add_argument('--init', '--init-method', dest='init_method', default='silhouettes', help="How to initialize vectors before sampling")
    parser.add_argument('--energy', action='store_true', help='Along with each sample generated, print its free energy')
    parser.add_argument('--every', type=int, default=None, help='Sample once every this many iters. Incompatible with --prog and --table.')

    args = parser.parse_args()

    args.init_method = VisInit[args.init_method]

    for model_fname in args.model_fname:
        print "Drawing samples from model defined at {}".format(model_fname)
        f = open(model_fname)
        model = pickle.load(f)
        f.close()
        # TODO: add as arg
        if 'usgeo' in model_fname:
            example_file = 'data/usgeo.txt'
        elif 'reponames' in model_fname:
            example_file = 'data/reponames.txt'
        elif 'names' in model_fname:
            example_file = 'data/names2.txt'

