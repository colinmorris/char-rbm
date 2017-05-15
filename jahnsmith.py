from energy_checker import energizer
import pickle
import bokeh.plotting as bplt
import json

CANON = ['john smith']
SINGLE_SWAPS = ['zohn smith', 'cohn smith', 'jahn smith', 'jonn smith', 'john smite', 'john svith']
OTHER = ['john john', 'smith smith', 'smith john', 'jon smith', 'zhong smith', 'zhong wang', 'john wang']

VARIATIONS = CANON + SINGLE_SWAPS + OTHER
    

with open("models/canon/actors.pickle") as f:
    model = pickle.load(f)

name_energy = {}
for mutant in VARIATIONS:
    name_energy[mutant] = energizer(model, mutant)

obj = {}
for glob in ['CANON', 'SINGLE_SWAPS', 'OTHER']:
    obj[glob.lower()] = {k:name_energy[k] for k in globals()[glob]}


with open('jahn.json', 'w') as f:
    json.dump(obj, f)

"""
bplt.output_file('color_scatter.html')

p = bplt.figure()
p.circle(name_energy.values(), [1 for _ in name_energy],
    fill_alpha=0.6)
bplt.show(p)
"""
