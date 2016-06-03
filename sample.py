import argparse
import pickle
import numpy as np
import utils
import enum

import common
from short_text_codec import ShortTextCodec

MAX_PROG_SAMPLE_INTERVAL = 10000

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

def print_samples(model, visibles):
    for v in visibles:
        print model.codec.decode(v, pretty=True)

class BadInitMethodException(Exception):
    pass

def starting_visible_configs(init_method, n, model, training_examples_fname):
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
    elif init_method == VisInit.chunks:
        # This works, but probably isn't idiomatic numpy.
        # I don't think I'll ever write idiomatic numpy.

        # Start w uniform dist
        char_indices = np.random.randint(0, nchars, (n,maxlen))
        # Choose some random lengths
        lengths = np.clip(maxlen*.25 * np.random.randn(n) + (maxlen*.66), 1, maxlen
            ).astype('int8').reshape(n, 1)
        _, i = np.indices((n, maxlen))
        char_indices[i>=lengths] = model.codec.char_lookup[model.codec.filler]
        
        return np.eye(nchars)[char_indices.ravel()].reshape(vis_shape)
    else:
        raise ValueError("Unrecognized init method: {}".format(init_method))


@common.timeit
def sample_model(model, n, iters, prog, max_prob, init_method=VisInit.biases, training_examples=None):
    vis = starting_visible_configs(init_method, n, model, training_examples)
    # #iters -> list of strings
    model_samples = {}
    power = 0
    i = 0
    def gather(visible):
        # Turn off 'strict' mode for i>0. The only way we'll have invalid one-hot vectors
        # past that point is if we trained our model without softmax sampling. If so, we
        # want to visualize the most likely char at each position.
        model_samples[i] = [model.codec.decode(v, pretty=True, strict=i==0) for v in visible]
    gather(vis) 
    while i < iters:
        if prog and (i == 10**power or i % MAX_PROG_SAMPLE_INTERVAL == 0) and i > 0:
            power += 1
            sample = model.gibbs(vis, sample_max=max_prob)
            gather(sample)
        vis = model.gibbs(vis)
        i += 1

    sample = model.gibbs(vis, sample_max=max_prob)
    gather(sample)
    return model_samples

# TODO: This probably belongs in visualize.py
def render_sample_table(samples, model, model_pickle_name):
    model_name = model_pickle_name.split('.')[0].split('/')[-1]
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
        th.rowhead {
            background: #eee;
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
    <body>''')
    s += '<h2>Model: {}</h2>'.format(model_name)
    s += '<table><thead>'
    inits = sorted(samples.keys())
    niters = sorted(samples[inits[0]].keys())
    s += '<tr><th class="top" colspan="{}"># Iterations</th></tr>'.format(len(niters)+1)
    
    header_cells = (['<th class="rowhead">Initialization Method</th>'] + 
        ['<th>{:,}{}</th>'.format(n, '*' if n==0 else '') for n in niters])
    s += '<tr>{}</tr></thead><tbody>'.format(''.join(header_cells))
    
    # Row for each init method
    for k in inits: 
        cells = []
        for n in niters:
            lis = ['<li>{}</li>'.format(sampled_string) for sampled_string in samples[k][n]]
            cell = '<td><ul>{}</ul></td>'.format(''.join(lis))
            cells.append(cell)
        s += '<tr><th class="rowhead">{}</th>{}</tr>'.format(k, ''.join(cells))
    s += '''</tbody></table>
    <aside>*Certain degenerate starting configurations of visible units (e.g. all zeros) can't be
    meaningfully represented as strings, because they don't consist of valid 'one-hot' vectors. 
   These malformed vectors are rendered as '<span style="font-family: monospace">?</span>'.</aside>
    </body></html>'''
    f = open('tablesamples_{}.html'.format(model_name), 'w')
    f.write(s)
    f.close()

def sample_table(model, example_file, args):
    samples = {}
    for init in reversed(VisInit):
        print "init = " + str(init)
        try:
            # Note: we ignore the --prog option, since table-mode implies we want to do progrssive sampling
            samples[init.name] = sample_model(model, args.n_samples, args.iters, 
                prog=True, max_prob=not args.nomax, init_method=init, 
                training_examples=example_file)
        except BadInitMethodException as e:
            print e
    render_sample_table(samples, model, model_fname)

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
    parser.add_argument('--table', action='store_true', help='Generate an html table of samples drawn at different ' +
                        'numbers of iterations starting with each possible initialization method. This presumes --prog ' +
                        'and ignores options relating to initialization method (since it cycles through them all).')
    parser.add_argument('--no-max', dest='nomax', action='store_true',
                        help='Default behaviour is to perform n rounds of Gibbs sampling, then one more ' +
                        'special round of sampling where we take the visible unit with the highest probability. ' +
                        'If this flag is enabled, the final round of sampling will be standard one, where we ' +
                        'sample randomly according to the softmax probabilities of visible units.')
    parser.add_argument('--init', '--init-method', dest='init_method', type=int, default=VisInit.silhouettes)

    args = parser.parse_args()

    for model_fname in args.model_fname:
        print "Drawing samples from model defined at {}".format(model_fname)
        f = open(model_fname)
        model = pickle.load(f)
        f.close()
        if 'usgeo' in model_fname:
            example_file = 'data/microgeo.txt'
        elif 'reponames' in model_fname:
            example_file = 'data/reponames.txt'
        elif 'names' in model_fname:
            example_file = 'data/names2.txt'

        if args.table:
            sample_table(model, example_file, args)
        else:
            samples = sample_model(model, args.n_samples, args.iters, args.prog, 
                                    max_prob=not args.nomax, init_method=args.init_method, 
                                    training_examples=example_file)
            if len(samples) == 1:
                print '\n'.join(samples.values()[0])
            else:
                for niters in sorted(samples.keys()):
                    print "After {} iterations...".format(niters)
                    print '\n'.join(samples[niters])
                    print
