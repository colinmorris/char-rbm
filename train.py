import argparse
import pickle

from sklearn.cross_validation import train_test_split

import common
from short_text_codec import ShortTextCodec
from rbm_softmax import CharBernoulliRBM, CharBernoulliRBMSoftmax


def pickle_name(args):
    fname = args.tag if args.tag else args.input_fname.split('.')[0].split('/')[-1]
    fname += '_'
    # TODO: For the sake of brevity, maybe we should only encode parameters that didn't take their default value (though defaults may drift over time...)
    if not args.softmax:
        fname += '_nosm'
    if args.learning_rate_backoff:
        fname += '_lrb'
    fname += '_{}_{}_{:.3f}.pickle'.format(args.epochs, args.n_hidden, args.learning_rate)
    return fname


if __name__ == '__main__':
    # TODO: An option for checkpointing model every n epochs
    parser = argparse.ArgumentParser(description='Train a character-level RBM on short texts',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_fname', metavar='txtfile',
                        help='A text file to train on, with one instance per line')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float, default=0.05,
                        help='The ratio of data to hold out to monitor for overfitting')
    parser.add_argument('--no-softmax', dest='softmax', action='store_false',
                        help='Don\'t use softmax visible units')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10,
                              help='Size of a (mini)batch. This also controls # of fantasy particles.')
    parser.add_argument('--len', dest='text_length', type=int, default=20,
                        help='Maximum length of strings (i.e. # of softmax units).' +
                        ' Longer lines in the input file will be ignored')
    # TODO: It'd be cool to be able to say "take the n most frequent non-alpha characters in the input file"
    parser.add_argument('--extra-chars', dest='extra_chars', default='',
                        help='Characters to consider in addition to [a-zA-Z ]')
    parser.add_argument('--hid', '--hidden-units', dest='n_hidden', default=180, type=int,
                        help='Number of hidden units')
    parser.add_argument('-l', '--learning-rate', dest='learning_rate', default=0.05, type=float)
    parser.add_argument('--lr-backoff', dest='learning_rate_backoff', action='store_true',
                        help='Gradually reduce the learning rate at each epoch')
    parser.add_argument('-e', '--epochs', dest='epochs', default=5, type=int, help="Number of times to cycle through the training data")
    parser.add_argument('-m', '--model', dest='model', default=None,
                        help="Start from a previously trained model. Options affecting network topology will be ignored.")
    parser.add_argument('--tag', dest='tag', default='',
                        help='A name for this run. The model will be pickled to ' +
                        'a corresponding filename. That name will already encode ' +
                        'important hyperparams.')

    args = parser.parse_args()

    # TODO: trap ctrl+c and pickle the model before bailing
    if args.model:
        f = open(args.model)
        rbm = pickle.load(f)
        f.close()
        rbm.learning_rate = args.learning_rate
        rbm.base_learning_rate = args.learning_rate
        rbm.lr_backoff = args.learning_rate_backoff
        rbm.n_iter = args.epochs
        rbm.batch_size = args.batch_size
        codec = rbm.codec
    else:
        codec = ShortTextCodec(args.extra_chars, args.text_length)
        model_kwargs = {'codec': codec,
                        'n_components': args.n_hidden,
                        'learning_rate': args.learning_rate,
                        'lr_backoff': args.learning_rate_backoff,
                        'n_iter': args.epochs,
                        'verbose': 1,
                        'batch_size': args.batch_size,
                        }
        kls = CharBernoulliRBMSoftmax if args.softmax else CharBernoulliRBM
        rbm = kls(**model_kwargs)

    vecs = common.vectors_from_txtfile(args.input_fname, codec)
    train, validation = train_test_split(vecs, test_size=args.test_ratio)
    print "Training data shape : " + str(train.shape)

    rbm.fit(train, validation)
    f = open(pickle_name(args), 'wb')
    pickle.dump(rbm, f)
    f.close()
