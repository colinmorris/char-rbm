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
    # Treat visible biases as softmax
    biases = 2
    # Turn on each unit (not just each one-hot vector) with p=.5
    uniform = 3
    spaces = 4
    # Training examples
    train = 5
    # TODO: Use training examples but randomly mutate non-space/padding characters

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
    # #iters -> list of strings
    model_samples = {}
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
    power = 0
    i = 0
    def gather(visible):
        model_samples[i] = [model.codec.decode(v, pretty=True) for v in visible]
    gather(vis) # XXX
    while i < iters:
        if prog and (i == 10**power):
            power += 1
            sample = model.gibbs(vis, sample_max=max_prob)
            gather(sample)
        vis = model.gibbs(vis)
        i += 1

    sample = model.gibbs(vis, sample_max=max_prob)
    gather(sample)
    return model_samples

# TODO: This probably belongs in visualize.py
def sample_table(samples, model):
    sorted_keys = sorted(samples.keys())
    header_cells = (['<th class="dummy"></th>'] + 
        ['<th>{}</th>'.format(s) for s in sorted_keys])
    s = ('''<html><head><style>
        table {
            background: #ccc;
            border-spacing: 1px;
            }
        td {
            font-family: monospace;
            }
        li {
            list-style-type: none;
            }
        th.dummy {
            background: #ccc;
            }
        th.top {
            font-size: larger;
            line-height: 2em;
            }
        td, th {
            background: #fff;
            padding: 5px;
            }
        th {
            padding-right: 1em;
            }
        ul {
            width: ''' + str(model.codec.maxlen+2) + '''ch;
            -webkit-padding-start: 1ch;
            }
    </style></head>
    <body>
        <table><thead>''' + 
            '<tr><th class="dummy"/><th class="top" colspan="{}">Initialization Method</th></tr>'.format(len(samples)) +
            '<tr>{}</tr></thead>'.format(''.join(header_cells)))
    # Row for each number of iterations
    for niters in sorted(samples[sorted_keys[0]]):
        cells = []
        for k in sorted_keys:
            lis = ['<li>{}</li>'.format(sampled_string) for sampled_string in samples[k][niters]]
            cell = '<td><ul>{}</ul></td>'.format(''.join(lis))
            cells.append(cell)
        s += '<tr><th>{:,} iters{}</th>{}</tr>'.format(niters, '*' if niters==0 else '', ''.join(cells))
    s += '''</table>
    <aside>*Certain degenerate starting configurations of visible units (e.g. all zeros) can't be
    meaningfully represented as strings, because they don't consist of valid 'one-hot' vectors. 
   These malformed vectors are rendered as '<span style="font-family: monospace">?</span>'.</aside>
    </body></html>'''
    f = open('table_of_samples.html', 'w')
    f.write(s)
    f.close()

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

    # TODO: Make initialization method configurable
    args = parser.parse_args()

    for model_fname in args.model_fname:
        print "Drawing samples from model defined at {}".format(model_fname)
        f = open(model_fname)
        model = pickle.load(f)
        f.close()
        samples = {}
        #for init in [VisInit.zeros, VisInit.uniform, VisInit.biases, VisInit.spaces]:
        for init in VisInit:
            print "init = " + str(init)
            samples[init.name] = sample_model(model, args.n_samples, args.iters, args.prog, max_prob=not args.nomax, init_method=init, training_examples='data/microgeo.txt')
        sample_table(samples, model)
