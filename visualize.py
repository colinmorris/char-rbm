"""These are pretty hacky, but they get the job done.
# TODO: It would be nice to have a command line visualization (at least for activation rate)
"""

import sys
import os
import pickle
import bokeh.plotting
import bokeh.io
import numpy as np

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

def visualize_hidden_activations(model, example_fname, out="activations.html"):
    bokeh.plotting.output_file(out, title="Hidden activations - {}".format(model.name))
    figures = []
    n = 300 # TODO: make configurable
    vecs = common.vectors_from_txtfile(example_fname, model.codec, limit=n) 
    for n_gibbs in [0, 5, 1000]:
        if n_gibbs > 0:
            vecs = model.repeated_gibbs(vecs, n_gibbs, sample_max=False)
        # TODO: Visualize hidden probabilities to avoid sampling noise? Should at least offer option
        hiddens = model._sample_hiddens(vecs)
        y, x = np.nonzero(hiddens)
        max_y, max_x = hiddens.shape
        hidden_counts = np.sum(hiddens, axis=0)
        n_dead = (hidden_counts == 0).sum()
        n_immortal = (hidden_counts == n).sum()
        p = bokeh.plotting.figure(title="After {} rounds of Gibbs sampling. Dead = {}. Immortal = {}".format(n_gibbs, n_dead, n_immortal),
                    x_axis_location="above", x_range=(0,hiddens.shape[1]), y_range=(0,hiddens.shape[0])
        )
        p.plot_width = 1100
        sidelen = p.plot_width / (max_x + 0.0)
        p.plot_height = int(p.plot_width * (max_y / (max_x + 0.0)))
        p.rect(x=x, y=y, width=sidelen, height=sidelen,
            width_units='screen', height_units='screen',
        )
        figures.append(p)

    p = bokeh.io.vplot(*figures)
    bokeh.plotting.save(p)


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
    model.name = os.path.basename(model_fname)

    tag = model_fname[:model_fname.rfind(".")]
    receptive_fields(model, tag + '_receptive_fields.html')

    visualize_hidden_activations(model, sys.argv[2], tag + '_activations.html')
