This directory contains a bunch of samples drawn from models trained on a few different datasets. `foo.txt` all the sampled names, internally deduped. `foo_unique.txt` is deduped against the corresponding training set. 

Details of each model are below.

# Actors

## Model

- 180 hidden units
- trained for 20 epochs with LR=.1 decaying linearly per epoch
- batch size = 20
- alphabet = `[a-z]$ .-`
- used specialized "binomial" codec (see `short_text_codec.py`)

## Sampling

Sampled with simulated annealing going from T=1.3 to T=0.3 over 800 iterations.

    python sample_every.py --dedupe -s 1.3 -n 4000 -e 0.3 -i 800 --energy $model 200

## Deduping

Out of 7,600 generated names...

- 5 exactly match a name in the training set
- 40 out of 775 distinct first names exist in the training set (n=1.5m)
- 32 out of 3,454 distinct last names exist in the training set

# US place names

## Model

- 350 hidden units
- 20 + 20 epochs on usgeo dataset with max length = 20
- lr=.05 in the first round, then .00 in second. decayed during each run.
- batch size 20 in first round then 40
- weight cost of 0.0001

## Sampling

Sampled with simulated annealing going from T=1.0 to T=0.3 over 2k iterations. Started from 'silhouettes' of training data (see `VisInit` enum in `sampling.py`).

    python sample_every.py --sil data/usgeo.txt --no-col --dedupe --energy -f 250 -s 1.0 -e 0.3 -i 2000 -n 10000 $model 250
    # dedupe and filter by energy < -145 

## Deduping

2,920 / 42,709 generated place names exist in the training set (n=700k).

# GitHub repositories

## Model

- 350 hidden units
- trained for 4 rounds of 20 epochs
- LR for final round = .001
- maxlen = 20, minlen = 6, alphabet is case sensitive and includes all special chars in dataset (nchars=66)

## Sampling

Sampled with simulated annealing going from T=1.5 to T=0.2 over 1k iterations. Particles initialized randomly according to biases of the visible units.

    python sample_every.py -s 1.5 -e 0.2 -i 1000 -f 400 --energy --no-col --dedupe -n 10000 $model 100

## Deduping

2,605 / 20,000 generated repo names exist in the training set (n=3.7m)

# Games

## Model

- 250 hidden units
- alphabet = `[a-z][0-9]$ !-:&.'`
- trained for 80+10 epochs

## Sampling

Sampled with simulated annealing going from T=1.0 to T=0.2 over 250 iterations. Started from 'silhouettes' of training data (see `VisInit` enum in `sampling.py`).

    python sample_every.py -n 10000 -f 100 -i 250 -s 1.0 -e 0.2 --sil data/games.txt $model 50

## Deduping

34 / 3,490 generated games exist in the training set (n=80k)
