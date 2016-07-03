import sampling
import pickle
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample short texts from a pickled model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_fname', metavar='model.pickle', nargs='+',
                        help='One or more pickled RBM models')
    parser.add_argument('-n', '--n-samples', dest='n_samples', type=int, default=1,
                              help='How many samples to draw')
    parser.add_argument('-i', '--iters', dest='iters', type=int, default=10**3,
                              help='How many rounds of Gibbs sampling to perform')
    parser.add_argument('--energy', action='store_true', help='Along with each sample generated, print its free energy')
    parser.add_argument('-s', '--start-temp', dest='start_temp', type=float, default=1.0)
    parser.add_argument('-e', '--end-temp', dest='end_temp', type=float, default=1.0)

    args = parser.parse_args()


    for model_fname in args.model_fname:
        if len(args.model_fname) > 1:
            print "Drawing samples from model defined at {}".format(model_fname)
        f = open(model_fname)
        model = pickle.load(f)
        model.intercept_hidden_[148] = 10
        f.close()

        sample_indices = [args.iters-1]
        vis = sampling.sample_model(model, args.n_samples, args.iters, sample_indices,
                                start_temp=args.start_temp, final_temp=args.end_temp,
                                sample_energy=args.energy)

        h = model._sample_hiddens(vis, temperature=0.01)
        for subarr in h:
            print np.nonzero(subarr)[0] + 1
