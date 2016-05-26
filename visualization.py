# ZZZ
_CHAR2I = {' ': 26}
for (i, o) in enumerate(range(ord('a'), ord('z')+1)):
  _CHAR2I[chr(o)] = i
for (i, o) in enumerate(range(ord('A'), ord('Z')+1)):
  _CHAR2I[chr(o)] = i
  
# This happens to work out such that we get the lowercase letters as values, which is nice.
_I2CHAR = {v:k for (k, v) in _CHAR2I.iteritems()}

def vectors_from_txtfile(fname):
  f = open(fname)
  skipped = 0
  vecs = []
  for line in f:
    line = line.strip()
    if len(line) > MAXLEN:
      skipped += 1
      continue
    try:
      vecs.append(vectorize_str(line))
    except KeyError:
      # Some non-ascii chars slipped in
      skipped += 1

  print "Gathered {} vectors. Skipped {}".format(len(vecs), skipped)
  # TODO: Why default to dtype=float? Seems wasteful? Maybe it doesn't really matter. Actually, docs here seem inconsistent? Constructor docs say default float. transform docs say int. 
  # TODO: should probably try using a sparse matrix here
  vecs = np.asarray(vecs)
  print vecs.shape
  return OneHotEncoder(NCHARS).fit_transform(vecs)

def receptive_fields(model):
  f = open('recep.html', 'w')
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
  UPPER_THRESH = THRESH*3
  def style(w):
    if w <= UPPER_THRESH:
      return "opacity: {:.2f}".format(w/UPPER_THRESH)
    return "font-size: {:.2f}em".format(w/UPPER_THRESH)
  opacity = lambda w: min(w, 1.0) / 1.0
  MAXLEN, NCHARS = model.softmax_shape
  for component_index, h in enumerate(model.components_):
    res += '<div><h2>' + str(component_index) + '</h2>'
    for cindex in range(MAXLEN):
      weights = zip(range(NCHARS), h[cindex*NCHARS:(cindex+1)*NCHARS])
      weights.sort(key = lambda w: w[1], reverse=True)
      # Highly positive weights
      res += '<span class="pos"><span class="chars">'
      for i, w in weights:
        if w < THRESH:
          break
        char = _I2CHAR[i]
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
        char = _I2CHAR[i]
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
  f.close()
      
def visualize_hidden_activations(model, example_fname):
  s = '''<html><head><style>
    body {
      font-family: monospace;
      font-size: 0.9em;
      }
    .n {
      color: black;
      opacity: .2;
      }
    .y {
      color: blue;
      }
  </style></head><body><pre>'''
  vecs = vectors_from_txtfile(example_fname)
  hiddens = model._sample_hiddens(vecs, check_random_state(model.random_state))
  PADDING = 3 + 1
  s += ' '*5 + '0'
  for i in range(5*PADDING, hiddens.shape[1]*PADDING, 5*PADDING):
    s += str(i/PADDING).rjust(5*PADDING, ' ')
  s += '<br/>'
  for i, hid in enumerate(hiddens):
    #s += '{:3d}  '.format(i) + ''.join(['|' if h == 1 else '.' for h in hid]) + '<br/>'
    s += '{:3d}  '.format(i) + ''.join(
      ['<span class="{}">|{}</span>'.format("y" if h == 1 else "n", " "*(PADDING-1)) for h in hid]
      )
    s += ' ' + str(sum(hid))
    s += '<br/>'
  
  
  s += ' ' * 5 + ''.join([str(sum(active)).ljust(PADDING, ' ') 
    for active in hiddens.T])
  s += '</pre></body></html>'
  fout = open('hidden_activations.html', 'w')
  fout.write(s)
