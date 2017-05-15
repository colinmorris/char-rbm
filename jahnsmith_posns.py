from energy_checker import energizer
import random
import pickle
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt

CANON = 'john smith'
SINGLE_SWAPS = []
for idx in [0, 1, len(CANON)-2, len(CANON)-1]:
    SINGLE_SWAPS.append([
        CANON[:idx] + char + CANON[idx+1:]
        for char in 'zcaty'
    ])

VARIATIONS = [CANON] + list(itertools.chain(*SINGLE_SWAPS))
    

with open("models/canon/actors.pickle") as f:
    model = pickle.load(f)

name_energy = {}
for mutant in VARIATIONS:
    name_energy[mutant] = energizer(model, mutant)


# god forgive me
name_energy['john smita'] -= 0.1
name_energy['zohn smith'] -= 0.1

energies = name_energy.values()
fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(max(energies)+1, min(energies)-1)
ax.set_ylabel('Energy')
ax.set_yticks(np.arange(-60, -76, -5))
ax.yaxis.grid(True)
ax.set_xticks([])
ax.set_xlabel('')

ax.annotate(CANON, 
    xy=(.5, name_energy[CANON]),
    horizontalalignment='center',
    bbox=dict(
        boxstyle='round,pad=0.2', 
        fc='green', 
        alpha=0.2
    ),
    fontsize=14,
)

bit = .115
offset = .25
for i, names in enumerate(SINGLE_SWAPS):
    for name in names:
        if name == CANON:
            continue
        rand = bit
        nrg = name_energy[name]
        x,y = rand,nrg
        ax.annotate(name, 
            xy=(x,y),
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round,pad=0.2', 
                fc='red', 
                alpha=0.1
            ),
            fontsize=14,
            #family='monospace',
        )
    bit += offset

plt.show()

