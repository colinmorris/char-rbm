from energy_checker import energizer
import random
import pickle
import json
import matplotlib.pyplot as plt

CANON = ['john smith']
SINGLE_SWAPS = ['zohn smith', 'cohn smith', 'jahn smith', 'jonn smith', 
                'john smite', 'john svith', 'john smhth',
                ]
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


energies = name_energy.values()
fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(min(energies)-1, max(energies)+1)

group_to_color = {
    'canon': 'green',
    'single_swaps': 'mediumspringgreen',
    'other': 'purple',
}

groupnos = {'canon':.5, 'single_swaps':.4, 'other':.6}
for group, names in obj.iteritems():
    for name in names:
        rand = groupnos[group]
        nrg = names[name]
        x,y = rand,nrg
        ax.annotate(name, 
            xy=(x,y),
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round,pad=0.2', 
                fc=group_to_color[group], 
                alpha=0.2
            ),
        )

plt.show()

