import Sampling
import sys
import pickle
import argparse
import colorama
colorama.init()

SAMPLES = []
def horizontal_cb(strings, i, energy=None):
    global SAMPLES
    if energy is not None:
        SAMPLES.append(zip(strings, energy))
    else:
        SAMPLES.append(strings)

DEDUPE_SEEN = []
def dedupe_cb(strings, i, energy=None):
    global DEDUPE_SEEN
    if not DEDUPE_SEEN:
        DEDUPE_SEEN = [set() for _ in strings]
    for i in range(len(strings)):
        if strings[i] in DEDUPE_SEEN[i]:
            continue
        print strings[i] + "\t" + ("{:.2f}".format(energy[i]) if energy is not None else "")
        DEDUPE_SEEN[i].add(strings[i])
    print

def bold(s):
    return "\033[31m" + s + "\033[0m"

def print_columns(maxlen):
    col_width = maxlen+2
    for fantasy_index in range(len(SAMPLES[0])):
        particles = [s[fantasy_index] for s in SAMPLES]
        if args.energy:
            min_energy = min(particles, key=lambda tup: tup[1])
            print "".join(
                bold(p[0].ljust(col_width)) if p == min_energy
                    else p[0].ljust(col_width)
                for p in particles)
        else:
            print "".join(s[fantasy_index].ljust(col_width) for s in SAMPLES) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample short texts from a pickled model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_fname', metavar='model.pickle', nargs='+',
                        help='One or more pickled RBM models')
    parser.add_argument('--every', type=int, default=-1, help='How often to sample.' +
                        ' If -1 (default) only sample after the last iteration.')
    parser.add_argument('-n', '--n-samples', dest='n_samples', type=int, default=30,
                              help='How many samples to draw')
    parser.add_argument('-f', '--first', dest='first', type=int, default=-1,
                              help='Which iteration to draw the first sample at ' +
                              '(if --every is provided and this is not, defaults to --every)')
    parser.add_argument('-i', '--iters', dest='iters', type=int, default=10**3,
                              help='How many rounds of Gibbs sampling to perform')
    parser.add_argument('--energy', action='store_true', help='Along with each sample generated, print its free energy')
    parser.add_argument('-s', '--start-temp', dest='start_temp', type=float, default=1.0, help="Temperature for first iteration")
    parser.add_argument('-e', '--end-temp', dest='end_temp', type=float, default=1.0, help="Temperature at last iteration")
    parser.add_argument('--no-col', dest='columns', action='store_false')
    parser.add_argument('--dedupe', action='store_true')
    parser.add_argument('--sil', help='data file for silhouettes')

    args = parser.parse_args()


    for model_fname in args.model_fname:
        if len(args.model_fname) > 1 or not args.columns:
            print "Drawing samples from model defined at {}".format(model_fname)
        f = open(model_fname)
        model = pickle.load(f)
        f.close()

        if args.every == -1:
            sample_indices = [args.iters-1]
        else:
            first = args.every if args.first == -1 else args.first
            sample_indices = range(first, args.iters, args.every)
            if sample_indices[-1] != args.iters - 1:
                sample_indices.append(args.iters-1)
        
        if args.columns:
            cb = horizontal_cb
        elif args.dedupe:
            cb = dedupe_cb
        else:
            cb = Sampling.print_sample_callback

        kwargs = dict(start_temp=args.start_temp, final_temp=args.end_temp, sample_energy=args.energy, 
                    callback=cb)
        if args.sil:
            kwargs['init_method'] = Sampling.VisInit.silhouettes
            kwargs['training_examples'] = args.sil
        
        vis = Sampling.sample_model(model, args.n_samples, args.iters, sample_indices, **kwargs)

        if args.columns:
            print_columns(model.codec.maxlen)

        if args.energy:
            fe = model._free_energy(vis)
            sys.stderr.write('Final energy: {:.2f} (stdev={:.2f})\n'.format(fe.mean(), fe.std()))

