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
import sampling

RECEPTIVE_MAX_CHARS_PER_INDEX = 5
RECEPTIVE_WEIGHT_PERCENTILE = 90

N_MOST_WANTED = 10
MOST_WANTED_MIN_PROB = .65

def trchar(char):
    if char == ' ':
        return '_'
    return char


def most_wanted(hidx, samples, mean_hiddens):
    top_indices = mean_hiddens[:,hidx].argsort()[-N_MOST_WANTED:][::-1]
    s = '<div class="mostwanted col-xs-4"><ol>'
    for idx in top_indices:
        prob = mean_hiddens[idx,hidx]
        if prob < MOST_WANTED_MIN_PROB:
            if idx == top_indices[0]:
                print "WARNING: no activations above threshold for hidden unit with index {}".format(hidx)
            break
        inline = ' style="opacity: {:.2f}"'.format(prob**4) if prob <= .97 else ''
        s += '<li{}>{} <span>({:.3f})</span></li>'.format(inline, samples[idx], prob)
    s += '</ol></div>'
    return s

def hidden_unit_table(model, hidden_index, weight_thresh, max_opacity):
    maxlen = model.codec.maxlen
    nchars = model.codec.nchars
    s = '<div class="activations col-xs-8"><table>'
    pro_chars = []
    con_chars = []

    pos_color = '100,250,100'
    neg_color = '250,100,100'

    def style(w):
        if w < 0:
            w = -1 * w
            c = neg_color
        else:
            c = pos_color
        delta = max_opacity - weight_thresh
        opacity = min(1.0, (w - weight_thresh)/delta)
        return 'style="background-color: rgba({}, {:.2f}"'.format(c, opacity)

    for string_index in range(maxlen):
        weights = zip(range(nchars), model.components_[hidden_index][string_index*nchars:(string_index+1)*nchars])
        weights.sort(key=lambda w: w[1], reverse=True)
        pro = []
        con = []
        for i in range(RECEPTIVE_MAX_CHARS_PER_INDEX):
            charindex, w = weights[i]
            if w >= weight_thresh:
                # TODO: weight css
                pro.append((w, trchar(model.codec.alphabet[charindex])))
            else:
                break

        for i in range(RECEPTIVE_MAX_CHARS_PER_INDEX):
            charindex, w = weights[-(i+1)]
            if w*-1 >= weight_thresh:
                # TODO: weight css
                con.append((w,trchar(model.codec.alphabet[charindex])))
            else:
                break

        pro_chars.append(pro)
        con_chars.append(con)

    for i in range(RECEPTIVE_MAX_CHARS_PER_INDEX):
        s += '<tr>'
        for pc in pro_chars:
            if len(pc) >= (i+1):
                w, c = pc[i]
                s += '<td {}>{}</td>'.format(style(w), c)
            else:
                s += '<td></td>'
        s += '</tr>'

    s += '<tr class="buffer"></tr>'

    for i in range(RECEPTIVE_MAX_CHARS_PER_INDEX):
        s += '<tr>'
        actual_index = RECEPTIVE_MAX_CHARS_PER_INDEX - 1 - i
        for cc in con_chars:
            if len(cc) > actual_index:
                w, c = cc[actual_index]
                s += '<td {}>{}</td>'.format(style(w), c)
            else:
                s += '<td></td>'
        s += '</tr>'
    s += '</table></div>'
    return s

def receptive_fields2(model, samples, out="recep.html"):
    mean_hiddens = model._mean_hiddens(samples)
    sample_strings = [model.codec.decode(s, pretty=True) for s in samples]
    f = open(out, 'w')
    weight_thresh = np.percentile(model.components_, RECEPTIVE_WEIGHT_PERCENTILE)
    max_opacity = np.percentile(model.components_, 100 - 0.1*(100 - RECEPTIVE_WEIGHT_PERCENTILE))
    f.write('''<html>
    <head><style>
        td {
            border-style: solid;
            border-width: thin;
            text-align: center;
            font-family: mono;
            width: 2em;
            height: 2em;
        }
        ol {
            font-size: large;
        }
        li>span {
            font-size: smaller;
            float: right;
        }
    </style>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous">
    </head><body><div class="container">
    ''')
    f.write('<h1>Hidden weights for model {}</h1>'.format(model.name))
    for hidden_index in range(model.components_.shape[0]):
        f.write('<div class="row unit"><h2>Hidden unit {}</h2>'.format(hidden_index+1))
        table = hidden_unit_table(model, hidden_index, weight_thresh, max_opacity)
        top_samples = most_wanted(hidden_index, sample_strings, mean_hiddens)
        f.write(table + top_samples)
        f.write("</div>")
    f.write('</div></body></html>')
    f.close()
    print "Wrote tables to {}".format(out)
        
        

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
            vecs = model.repeated_gibbs(vecs, n_gibbs)
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
    
    nvecs = 5000
    print "Loading vectors"
    vecs = common.vectors_from_txtfile(sys.argv[2], model.codec, limit=nvecs)
    print "Performing gibbs sampling"
    vis = sampling.sample_model(model, nvecs, 100, [499], 1.0, 0.2, None,
        starting_vis = vecs)
    
    print "Creating visualization"
    receptive_fields2(model, vis, tag + '_receptive_fields.html')
    
    # receptive_fields(model, tag + '_receptive_fields.html')

    # visualize_hidden_activations(model, sys.argv[2], tag + '_activations.html')
