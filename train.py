import argparse
import pickle

from sklearn.cross_validation import train_test_split

import common
from short_text_codec import ShortTextCodec
from rbm_softmax import CharBernoulliRBM, CharBernoulliRBMSoftmax

def pickle_name(args):
	fname = args.tag if args.tag else args.input_fname.split('.')[0]
	fname += '_'
	if not args.softmax:
		fname += '_nosm'
	fname += '_{}_{}_{:.3f}.pickle'.format(args.epochs, args.n_hidden, args.learning_rate)
	return fname
	

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a character-level RBM on short texts')
  parser.add_argument('input_fname', metavar='txtfile',
                      help='A text file to train on, with one instance per line')
  parser.add_argument('--test-ratio', dest='test_ratio', type=float, default=0.05,
                      help='The ratio of data to hold out to monitor for overfitting')
  parser.add_argument('--no-softmax', dest='softmax', action='store_false',
                      help='Don\'t use softmax visible units')
  parser.add_argument('--len', dest='text_length', type=int, default=20,
	                    help='Maximum length of strings (i.e. # of softmax units).' +
	                    ' Longer lines in the input file will be ignored')
  # TODO: It'd be cool to be able to say "take the n most frequent non-alpha characters in the input file"
  parser.add_argument('--extra-chars', dest='extra_chars', default='',
		                  help='Characters to consider in addition to [a-zA-Z ]')
  parser.add_argument('--hid', '--hidden-units', dest='n_hidden', default=180, type=int,
		                  help='Number of hidden units')
  parser.add_argument('-l', '--learning-rate', dest='learning_rate', default=0.05, type=float)
  parser.add_argument('-e', '--epochs', dest='epochs', default=5, type=int)
  parser.add_argument('--tag', dest='tag', default='',
										  help='A name for this run. The model will be pickled to ' +
										  'a corresponding filename. That name will already encode ' +
										  'important hyperparams.')

  args = parser.parse_args()

  # TODO: trap ctrl+c and pickle the model before bailing
  codec = ShortTextCodec(args.extra_chars, args.text_length)
  vecs = common.vectors_from_txtfile(args.input_fname, codec)
  train, validation = train_test_split(vecs, test_size=args.test_ratio)
  print "Training data shape : " + str(train.shape)
  model_kwargs = {'codec':codec,
	  'n_components': args.n_hidden,
	  'learning_rate': args.learning_rate,
	  'n_iter': args.epochs,
	  'verbose': 1
  }
  kls = CharBernoulliRBMSoftmax if args.softmax else CharBernoulliRBM
  rbm = kls(**model_kwargs)

  rbm.fit(train, validation)
  f = open(pickle_name(args), 'wb')
  pickle.dump(rbm, f)
  f.close()
