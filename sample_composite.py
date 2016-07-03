import sampling
import pickle
import argparse
from composite_rbm import CompositeRBM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample short texts from a pickled model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_fname', metavar='model.pickle', nargs='+',
                        help='One or more pickled RBM models')
    parser.add_argument('-n', '--n-samples', dest='n_samples', type=int, default=30,
                              help='How many samples to draw')
    parser.add_argument('-i', '--iters', dest='iters', type=int, default=10**3,
                              help='How many rounds of Gibbs sampling to perform')
    parser.add_argument('--energy', action='store_true', help='Along with each sample generated, print its free energy')
    parser.add_argument('-s', '--start-temp', dest='start_temp', type=float, default=1.0)
    parser.add_argument('-e', '--end-temp', dest='end_temp', type=float, default=1.0)

    args = parser.parse_args()

    models = []
    for model_fname in args.model_fname:
        f = open(model_fname)
        model = pickle.load(f)
        f.close()
        models.append(model)
    
    composite = CompositeRBM(models)

    sample_indices = [args.iters-1]
    
    sampling.sample_model(composite, args.n_samples, args.iters, sample_indices,
                                start_temp=args.start_temp, final_temp=args.end_temp,
                                sample_energy=args.energy)
