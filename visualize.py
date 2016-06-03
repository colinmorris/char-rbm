"""These are pretty hacky, but they get the job done.
# TODO: It would be nice to have a command line visualization (at least for activation rate)
"""

import sys
import pickle

import common


def receptive_fields(model, out="recep.html"):
    f = open(out, 'w')
    res = '''<html><head><style>
    span {
      padding-right: 10px;
    }
    span.chars {
      display: inline-block;
      min-width: 200px;
      min-height: 1.0em;
      }
    span.neg {
      color: maroon;
      }
    span.neg, span.pos {
      display: inline-block;
      width: 300px;
      }
  </style>
  </head>
  <body>
  '''
    THRESH = 1.5
    UPPER_THRESH = THRESH * 3

    def style(w):
        if w <= UPPER_THRESH:
            return "opacity: {:.2f}".format(w / UPPER_THRESH)
        return "font-size: {:.2f}em".format(w / UPPER_THRESH)
    opacity = lambda w: min(w, 1.0) / 1.0
    maxlen, nchars = model.codec.maxlen, model.codec.nchars
    for component_index, h in enumerate(model.components_):
        res += '<div><h2>' + str(component_index) + '</h2>'
        for cindex in range(maxlen):
            weights = zip(range(nchars), h[cindex * nchars:(cindex + 1) * nchars])
            weights.sort(key=lambda w: w[1], reverse=True)
            # Highly positive weights
            res += '<span class="pos"><span class="chars">'
            for i, w in weights:
                if w < THRESH:
                    break
                char = model.codec.alphabet[i]
                if char == ' ':
                    char = '_'
                res += '<span style="{}">'.format(style(w)) + char + '</span>'
            res += '</span>'
            maxw = weights[0][1]
            if maxw >= THRESH:
                res += '<span class="maxw">{:.1f}</span>'.format(weights[0][1])
            res += '</span>'

            # Highly negative weights
            res += '<span class="neg"><span class="chars">'
            for i, w in reversed(weights):
                w = -1 * w
                if w < THRESH:
                    break
                char = model.codec.alphabet[i]
                if char == ' ':
                    char = '_'
                res += '<span style="{}">'.format(style(w)) + char + '</span>'
            res += '</span>'
            minw = weights[-1][1] * -1
            if minw >= THRESH:
                res += '<span class="maxw">{:.1f}</span>'.format(minw)
            res += '</span>'

            res += '<br/>'
        res += '</div>'
    res += '</body></html>'
    f.write(res)
    print "Wrote visualization to " + out
    f.close()


def _hidden_activations_html(hiddens, title):
    # TODO: There are a million trillion better ways to do this. Could at least use svg.
    s = '<h2>{}</h2><pre>'.format(title)
    PADDING = 3 + 1
    s += ' ' * 5 + '0'
    for i in range(5 * PADDING, hiddens.shape[1] * PADDING, 5 * PADDING):
        s += str(i / PADDING).rjust(5 * PADDING, ' ')
    s += '<br/>'
    for i, hid in enumerate(hiddens):
        #s += '{:3d}  '.format(i) + ''.join(['|' if h == 1 else '.' for h in hid]) + '<br/>'
        s += '{:3d}  '.format(i) + ''.join(
            ['<span class="{}">|{}</span>'.format("y" if h == 1 else "n", " " * (PADDING - 1)) for h in hid]
        )
        s += ' ' + str(sum(hid))
        s += '<br/>'

    s += ' ' * 5 + ''.join([str(sum(active)).ljust(PADDING, ' ')
                            for active in hiddens.T])
    s += '</pre>'
    return s

'''
   for n_gibbs in [0, 5, 1000]:
        if n_gibbs > 0:
            vecs = model.repeated_gibbs(vecs, n_gibbs, sample_max=False)
        # TODO: Visualize hidden probabilities to avoid sampling noise? Should at least offer option
        hiddens = model._sample_hiddens(vecs)
        s += _hidden_activations_html(hiddens, 'Hidden activations after {} rounds of gibbs sampling'.format(n_gibbs))
    s += '</body></html>'
    fout = open(out, 'w')
    fout.write(s)
    print "Wrote visualization to " + out
'''
from bokeh.plotting import figure, output_file, show, save
import numpy as np
def visualize_hidden_activations(model, example_fname, out="activations.html"):
    output_file(out, title="Hidden activations")
    vecs = common.vectors_from_txtfile(example_fname, model.codec, limit=300) # TODO: make configurable
    for n_gibbs in [0, 5, 1000]:
        if n_gibbs > 0:
            vecs = model.repeated_gibbs(vecs, n_gibbs, sample_max=False)
        # TODO: Visualize hidden probabilities to avoid sampling noise? Should at least offer option
        hiddens = model._sample_hiddens(vecs)
        x, y = np.nonzero(hiddens)
        p = figure(title="After {} rounds of Gibbs sampling".format(n_gibbs),
                    x_axis_location="above", x_range=(0,hiddens.shape[1]), y_range=(0,hiddens.shape[0])
        )
        p.rect(x=x, y=y)
        save(p)
        break


if __name__ == '__main__':
    # TODO: argparse
    if len(sys.argv) < 3:
        print "USAGE: visualize.py model.pickle sample.txt"
        print (" (The sample file is used for visualizing the"
               + " activation rate of hidden units on typical inputs. It should be " +
               "no more than a few hundred lines")
        sys.exit(1)
    model_fname = sys.argv[1]
    f = open(model_fname)
    model = pickle.load(f)

    tag = model_fname[:model_fname.rfind(".")]
    receptive_fields(model, tag + '_receptive_fields.html')

    visualize_hidden_activations(model, sys.argv[2], tag + '_activations.html')
