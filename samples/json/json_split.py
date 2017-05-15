import sys
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("fname")
parser.add_argument("thresholds", nargs="+", type=float)

args = parser.parse_args()

threshes = sorted(args.thresholds)


f = open(args.fname)
skipped = 0
names = [ list() for _ in range(len(threshes)+1) ]
for line in f:
    try:
        if 'actors' in args.fname and 'first' not in args.fname:
            n1, n2, nrg = line.strip().split('\t')
            name = n1 + ' ' + n2
        else:
            name, nrg = line.strip().split('\t')
    except ValueError:
        skipped += 1
        continue
    nrg = float(nrg)

    for i, thresh in enumerate(threshes):
        if nrg < thresh:
            names[i].append(name)
            break
    else: # deal with it
        names[-1].append(name)

f.close()

fname = args.fname.split(os.path.sep)[-1]
fout = open(fname.split('.')[0] + '.json', 'w')
json.dump({'names':names}, fout)
fout.close()

print "Skipped {} lines".format(skipped)

print "Lines per threshold (coolest to warmest)..."
for i, n in enumerate(names):
    print '{}\t{}'.format(i, len(n))
