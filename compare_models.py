import argparse
import pickle
import common
import csv

common.DEBUG_TIMING = True

# TODO: Hyperparams would be nice. Models should probably just know their hyperparams.
FIELDS = ['name', 'nchars', 'minlen', 'maxlen', 'nhidden', 'PR_nudge', 'PR_noise', 'PR_sil', 'recon_error']

@common.timeit
def eval_model(model, trainfile, n):
    # TODO: Instead of just mean PR, should maybe also have stdev and median?
    row  = {'name': model.name[:6]}
    codec = model.codec
    row['nchars'] = codec.nchars
    row['minlen'] = getattr(codec, 'minlen', 0)
    row['maxlen'] = codec.maxlen
    row['nhidden'] = model.intercept_hidden_.shape[0]


    # The untainted vectorizations
    good = common.vectors_from_txtfile(trainfile, codec, n)
    bad = common.vectors_from_txtfile(trainfile, codec, n, codec.mutagen_nudge)
    row['PR_nudge'] = model.pseudolikelihood_ratio(good, bad)
    bad = common.vectors_from_txtfile(trainfile, codec, n, codec.mutagen_silhouettes)
    row['PR_sil'] = model.pseudolikelihood_ratio(good, bad)
    bad = common.vectors_from_txtfile(trainfile, codec, n, codec.mutagen_noise)
    row['PR_noise'] = model.pseudolikelihood_ratio(good, bad)

    # TODO
    row['recon_error'] = 1.0
    for k in row:
        if isinstance(row[k], float):
            row[k] = '{:.2f}'.format(row[k])
    return row

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', metavar='model', nargs='+', help='Pickled RBM models')
    parser.add_argument('trainfile', help='File with training examples')
    parser.add_argument('-n', type=int, default=10**4, help="Number of samples to average over")
    args = parser.parse_args()

    models = []
    for fname in args.models:
        f = open(fname)
        models.append(pickle.load(f))
        models[-1].name = fname
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
        row = eval_model(model, args.trainfile, args.n)
        writer.writerow(row)

    f.close()
    print "Wrote results to " + outname
