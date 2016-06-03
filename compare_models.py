import argparse
import pickle
import common
import csv
import os

common.DEBUG_TIMING = True

FIELDS = (['nchars', 'minlen', 'maxlen', 'nhidden',]
            + ['{}_{}'.format(metric, mut) for metric in ('PR', 'Err') 
                for mut in ('nudge', 'sil', 'noise')]
            + ['recon_error', 'filler', 'name']
            )

@common.timeit
def eval_model(model, trainfile, n):
    # TODO: Instead of just mean PR, should maybe also have stdev and median?
    row  = {'name': model.name}
    codec = model.codec
    row['nchars'] = codec.nchars
    row['minlen'] = getattr(codec, 'minlen', None)
    row['maxlen'] = codec.maxlen
    row['nhidden'] = model.intercept_hidden_.shape[0]
    row['filler'] = codec.filler

    # The untainted vectorizations
    good = common.vectors_from_txtfile(trainfile, codec, n)
    good_energy = model._free_energy(good)
    for name, mutagen in [ ('nudge', codec.mutagen_nudge), 
                            ('sil', codec.mutagen_silhouettes),
                            ('noise', codec.mutagen_noise),
                            ]:
        bad = common.vectors_from_txtfile(trainfile, codec, n, mutagen)
        bad_energy = model._free_energy(bad)

        # Pseudolikelihood ratio
        row['PR_{}'.format(name)] = (bad_energy - good_energy).mean()
        # "Error rate" (how often is lower energy assigned to the evil twin)
        row['Err_{}'.format(name)] = 100 * (bad_energy < good_energy).sum() / float(n)
        

    # TODO
    row['recon_error'] = 1.0
    for k in row:
        if isinstance(row[k], float):
            row[k] = '{:.1f}'.format(row[k])
        # By default, None is rendered as empty string, which messes up column output
        elif row[k] is None:
            row[k] = 'NA'
        elif row[k] == '':
            row[k] = "''"
    return row

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('models', metavar='model', nargs='+', help='Pickled RBM models')
    parser.add_argument('trainfile', help='File with training examples')
    parser.add_argument('-n', type=int, default=10**4, help="Number of samples to average over." +
                        "Default is pretty fast and, anecdotally, seems to give pretty reliable results."
                        + " Increasing it by a factor of 5-10 doesn't change much.")
    args = parser.parse_args()

    if args.trainfile.endswith('.pickle'):
        print "trainfile is mandatory"
        parser.print_usage()

    models = []
    for fname in args.models:
        f = open(fname)
        models.append(pickle.load(f))
        models[-1].name = os.path.basename(fname)
        f.close()

    # We could try to be efficient and only load the training data once for all models
    # But then we would need to require that all models passed in use equivalent codecs
    # Or do something clever to only load n times for n distinct codecs
    # Let's just do the dumb thing for now
    outname = 'model_comparison.csv'
    f = open(outname, 'w')
    writer = csv.DictWriter(f, FIELDS, delimiter='\t')
    writer.writeheader()

    for model in models:
        print "Evaluating " + str(model.name)
        row = eval_model(model, args.trainfile, args.n)
        writer.writerow(row)
        print

    f.close()
    print "Wrote results to " + outname
