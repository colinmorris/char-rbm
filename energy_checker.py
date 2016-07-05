import argparse
import pickle
import sys
from short_text_codec import BinomialShortTextCodec

def energizer(model, s):
    if isinstance(model.codec, BinomialShortTextCodec):
        parts = s.split(' ')
        first = parts[0]
        last = parts[1] if len(parts) == 2 else ''
        assert len(parts) <= 2, "wtf is this?"
        s = last + model.codec.separator + first
    vec = [model.codec.encode_onehot(s)]
    nrg = model._free_energy(vec)
    assert nrg.shape == (1,)
    return nrg[0]

parser = argparse.ArgumentParser(description="Check the energy a model assigns to some strings, interactively or by passing them on the command line")
parser.add_argument('model_file')
parser.add_argument('string', nargs='*')
parser.add_argument('-i', '--interactive', action='store_true')
args = parser.parse_args()

with open(args.model_file) as f:
    model = pickle.load(f)

for s in args.string:
   print "E({}) = {:.2f}".format(repr(s), energizer(model, s))

if args.interactive or len(args.string) == 0:
    print "Get energies of some strings interactively! q to quit"
    while 1:
        s = raw_input("Get energy of: ")
        if s.lower() == 'q':
            break
        print "{:.2f}".format(energizer(model, s))


