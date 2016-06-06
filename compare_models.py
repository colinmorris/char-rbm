import argparse
import pickle
import common
import csv
import os
import sklearn.metrics.pairwise
from sklearn.utils.extmath import log_logistic

common.DEBUG_TIMING = True

FIELDS = (['nchars', 'minlen', 'maxlen', 'nhidden', 'batch_size', 'epochs',]
            + ['pseudol9']
            + ['{}_{}'.format(metric, mut) for metric in ('Err',) 
                for mut in ('nudge', 'sil', 'noise')]
            + ['recon_error', 'mix_20', 'mix_200', 'filler', 'name']
            )

@common.timeit
def eval_model(model, trainfile, n):
    row  = {'name': model.name}
    # Comparing models with different codec params seems problematic when they change the 
    # set of examples each model is looking at (some strings will be too short/long for one
    # model but not another). This could introduce a systematic bias where some models get
    # strings that are a little easier or harder. Quick experiment performed to clamp minlen
    # and maxlen to a shared middle ground for all models. Didn't really affect ranking.
    codec = model.codec
    row['nchars'] = codec.nchars
    row['minlen'] = getattr(codec, 'minlen', None)
    row['maxlen'] = codec.maxlen
    row['nhidden'] = model.intercept_hidden_.shape[0]
    row['filler'] = codec.filler
    row['batch_size'] = model.batch_size
    # TODO: Not accurate for incrementally trained models
    row['epochs'] = model.n_iters


    # The untainted vectorizations
    good = common.vectors_from_txtfile(trainfile, codec, n)
    good_energy = model._free_energy(good)
    row['pseudol9'] = model.score_samples(good).mean()
    for name, mutagen in [ ('nudge', codec.mutagen_nudge), 
                            ('sil', codec.mutagen_silhouettes),
                            ('noise', codec.mutagen_noise),
                            ]:
        bad = common.vectors_from_txtfile(trainfile, codec, n, mutagen)
        bad_energy = model._free_energy(bad)

        # log-likelihood ratio
        # This is precisely log(P_model(good)/P_model(bad))
        # i.e. according to the model, how much more likely is the authentic data compared to the noised version?
        # Which seems like a really useful thing to know, but actually gives results that are pretty counterintuitive.
        # Some models score *really* well under this metric, but very poorly on the 'error rate' metric below and
        # on pseudo-likelihood. In fact, this metric seems to be inversely correlated with success on other metrics.
        # It's not clear to me why this is. My vague hypothesis is that models unconstrained by weight costs have
        # learned to associate really-really-really high (relative) energy to certain configurations. So the good
        # models 'win' more often (assigning lower energy to authentic examples), but the bad models sometimes win
        # by a lot more. (And for our purposes, maybe this shouldn't really be worth many more points. We just want
        # the strings we get from sampling to be reasonable, and for unreasonable strings to have high enough energy
        # that we won't encounter them. Whether "asdasdsf" gets HIGH_ENERGY, or HIGH_ENERGY x 10^100 doesn't really
        # matter to us.)
        # The connection to KL-divergence here is interesting. If we say P is the model distribution over the training
        # data and Q is the corresponding distribution which 'sees' the noised version (i.e. Q(v) := P(mutate(v))) 
        # then D_KL(P||Q) = \sum{v} P(v) * (energy(mutate(v)) - energy(v))
        # So our log-likelihood ratio is identical to KL-divergence except for the P(v) term (which is of course intractable).
        # The 'beating a dead horse' hypothesis is consistent with the 'bad' models having low KL-divergence in
        # spite of having a better log-likelihood ratio. These models may be assigning low absolute probabilities
        # to the training examples, but even lower probabilities to the mutants. So -log(P(v)) = 10^10^100, -log(P(mutate(v))) =
        # 10^10^200 isn't worth that many points, because the large ratio is tempered by the low P(v).
        # One could hypothesize that the opposite phenomenon is occurring, and the bad models are overfit to the 
        # training data, and have learned to assign precisely those strings very low energy. But using test data
        # (or even a different dataset similar to the training data - e.g. testing on Canadian geo names models trained
        # on US geo names) results in the same rankings.
        #row['LR_{}'.format(name)] = (bad_energy - good_energy).mean()
        
        # "Error rate" (how often is lower energy assigned to the evil twin)
        # TODO: Connection to noise contrastive estimation?
        row['Err_{}'.format(name)] = 100 * (bad_energy < good_energy).sum() / float(n)
        
    goodish = model.gibbs(good)
    row['recon_error'] = sklearn.metrics.pairwise.paired_distances(good, goodish).mean()
    goodisher = model.repeated_gibbs(good, 20, sample_max=False)
    row['mix_20'] = sklearn.metrics.pairwise.paired_distances(good, goodisher).mean()
    # TODO: This is too slow. Like 20 minutes per model.
    #goodish = model.repeated_gibbs(goodisher, 200, sample_max=False)
    row['mix_200'] = 0.0 #sklearn.metrics.pairwise.paired_distances(goodisher, goodish).mean()
    for k in row:
        if isinstance(row[k], float):
            row[k] = '{:.1f}'.format(row[k])
        # By default, None is rendered as empty string, which messes up column output
        elif row[k] is None:
            row[k] = 'NA'
        elif row[k] == '':
            row[k] = "''"
        elif row[k] == ' ':
            row[k] = "<sp>"
    return row

if __name__ == '__main__':
    # TODO: "Append" mode so we don't have to do a bunch of redundant calculations when we add one or two new models
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('models', metavar='model', nargs='+', help='Pickled RBM models')
    parser.add_argument('trainfile', help='File with training examples')
    parser.add_argument('-a', '--append', action='store_true', help='If there exists a model_comparison_<tag>.csv'
                        + ' file, append a row for each model file passed in, rather than clobbering. Does not '
                        + 'attempt to dedupe rows.')
    parser.add_argument('-t', '--tag', default='', help='A tag to append to the output csv filename')
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
    outname = 'model_comparison_{}.csv'.format(args.tag)
    append = args.append
    if append and not os.path.exists(outname):
        print "WARNING: received append option, but found no existing file {}".format(outname)
        append = False
    f = open(outname, 'a' if append else 'w')
    writer = csv.DictWriter(f, FIELDS, delimiter='\t')
    if not append:
        writer.writeheader()

    for i, model in enumerate(models):
        print "Evaluating {} [{}/{}]".format(model.name, i+1, len(models))
        row = eval_model(model, args.trainfile, args.n)
        writer.writerow(row)
        print

    f.close()
    print "Wrote results to " + outname
