import argparse
import pickle

from sklearn.cross_validation import train_test_split

import Utils
from ShortTextCodec import ShortTextCodec, BinomialShortTextCodec
from RBM import CharBernoulliRBM, CharBernoulliRBMSoftmax

def stringify_param(name, value):
    if name == 'tag':
        prefix = ''
    else:
        prefix = ''.join([token[0] for token in name.split('_')])

    if isinstance(value, bool):
        value = '' # The prefix alone tells us what we need to know - that this boolean param is the opposite of its default
    elif isinstance(value, float):
        # e.g. 1E-03
        value = '{:.0E}'.format(value)
    elif not isinstance(value, int) and not isinstance(value, basestring):
        raise ValueError("Don't know how to format {}".format(type(value)))
    return prefix + str(value)

def pickle_name(args, parser):
    fname = args.input_fname.split('.')[0].split('/')[-1]
    fname += '_'
    for arg in ['tag', 'batch_size', 'n_hidden', 'softmax', 'learning_rate_backoff', 'preserve_case', 'epochs', 'learning_rate', 'weight_cost', 'left']:
        value = getattr(args, arg)
        if value != parser.get_default(arg):
            fname += '_' + stringify_param(arg, value)

    return fname + '.pickle'


if __name__ == '__main__':
    # TODO: An option for checkpointing model every n epochs
    # TODO: Should maybe separate out vectorization and training? They're sort of
    # orthogonal (options like maxlen, preserve-case etc. don't even do anything
    # when starting from a pretrained model), and the options here are getting
    # bloated. 
    parser = argparse.ArgumentParser(description='Train a character-level RBM on short texts',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_fname', metavar='txtfile',
                        help='A text file to train on, with one instance per line')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float, default=0.05,
                        help='The ratio of data to hold out to monitor for overfitting')
    parser.add_argument('--no-softmax', dest='softmax', action='store_false',
                        help='Don\'t use softmax visible units')
    parser.add_argument('--preserve-case', dest='preserve_case', action='store_true',
                        help="Preserve case, rather than lowercasing all input strings. Increases size of visible layer substantially.")
    parser.add_argument('--binomial', action='store_true', help='Use the binomial text codec (for comma-separated two-part names)')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=10,
                              help='Size of a (mini)batch. This also controls # of fantasy particles.')
    parser.add_argument('--maxlen', dest='max_text_length', type=int, default=20,
                        help='Maximum length of strings (i.e. # of softmax units).' +
                        ' Longer lines in the input file will be ignored')
    parser.add_argument('--minlen', dest='min_text_length', type=int, default=0,
                        help='Minimum length of strings. Shorter lines in input file will be ignored.')
    # TODO: It'd be cool to be able to say "take the n most frequent non-alpha characters in the input file"
    parser.add_argument('--extra-chars', dest='extra_chars', default=' ',
                        help='Characters to consider in addition to [a-zA-Z]')
    parser.add_argument('--hid', '--hidden-units', dest='n_hidden', default=180, type=int,
                        help='Number of hidden units')
    parser.add_argument('-l', '--learning-rate', dest='learning_rate', default=0.1, type=float, help="Learning rate.")
    parser.add_argument('--weight-cost', dest='weight_cost', default=0.0001, type=float,
                        help='Multiplied by derivative of L2 norm on weights. Practical Guide recommends 0.0001 to start')
    parser.add_argument('--lr-backoff', dest='learning_rate_backoff', action='store_true',
                        help='Gradually reduce the learning rate at each epoch')
    parser.add_argument('-e', '--epochs', dest='epochs', default=5, type=int, help="Number of times to cycle through the training data")
    parser.add_argument('--left', action='store_true', help='Pad strings shorter than maxlen from the left rather than the right.')
    parser.add_argument('-m', '--model', dest='model', default=None,
                        help="Start from a previously trained model. Options affecting network topology will be ignored.")
    parser.add_argument('--tag', dest='tag', default='',
                        help='A name for this run. The model will be pickled to ' +
                        'a corresponding filename. That name will already encode ' +
                        'important hyperparams.')

    args = parser.parse_args()

    # TODO: trap ctrl+c and pickle the model before bailing

    # If the path to a pretrained, pickled model was provided, resurrect it, and
    # update the attributes that make sense to change (stuff like #hidden units,
    # or max string length of course can't be changed)
    if args.model:
        f = open(args.model)
        rbm = pickle.load(f)
        f.close()
        rbm.learning_rate = args.learning_rate
        rbm.base_learning_rate = args.learning_rate
        rbm.lr_backoff = args.learning_rate_backoff
        rbm.n_iter = args.epochs
        rbm.batch_size = args.batch_size
        rbm.weight_cost = args.weight_cost
        codec = rbm.codec
    else:
        codec_kls = BinomialShortTextCodec if args.binomial else ShortTextCodec
        codec = codec_kls(args.extra_chars, args.max_text_length, 
            args.min_text_length, args.preserve_case, args.left)
        model_kwargs = {'codec': codec,
                        'n_components': args.n_hidden,
                        'learning_rate': args.learning_rate,
                        'lr_backoff': args.learning_rate_backoff,
                        'n_iter': args.epochs,
                        'verbose': 1,
                        'batch_size': args.batch_size,
                        'weight_cost': args.weight_cost,
                        }
        kls = CharBernoulliRBMSoftmax if args.softmax else CharBernoulliRBM
        rbm = kls(**model_kwargs)

    vecs = Utils.vectors_from_txtfile(args.input_fname, codec)
    train, validation = train_test_split(vecs, test_size=args.test_ratio)
    print "Training data shape : " + str(train.shape)

    rbm.fit(train, validation)
    out_fname = pickle_name(args, parser)
    f = open(out_fname, 'wb')
    pickle.dump(rbm, f)
    f.close()
    print "Wrote model to " + out_fname
