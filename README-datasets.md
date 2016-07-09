# Datasets

## A note on preprocessing

In my experiments, I removed duplicates as part of the preprocessing step for each dataset below. I didn't think much of it at the time, but if I were to do these experiments again, I would *not* dedupe. The main argument against deduping is that it's giving the model a censored version of the density function you're trying to get it to learn. Also, as you change the amount of data you collect for your training set, you're changing the actual shape of the distribution. As you collect more data, the modes of your data will form increasingly smaller proportions of the dataset. In the limit, you would approach the uniform distribution. This concern is not entirely theoretical - for example, there are many GitHub repositories that seem to have been named completely randomly (example from the training set: `LlHZRbrhqYXMlX`).

On the other hand, having regions of extremely high probability can hurt your model's mixing rate and make sampling more difficult, so it's not an obvious decision.

## Usgeo

To download the geonames corpus of US geographical names:
    
    wget http://download.geonames.org/export/dump/US.zip
    unzip US.zip
    cut -f 2 US.txt > usnames.txt

That should give you around 2.2m names.

I filtered out punctuation and numerals in the beginning to make the problem as easy as possible, but empirically, adding a few more characters doesn't slow down training that much, and doesn't seem to hurt sample quality.

For more information on the geonames data, and to download names from other countries, check out the [geonames website](http://www.geonames.org/export/).

## Actors

I used [actors.list.gz](ftp://ftp.fu-berlin.de/pub/misc/movies/database/actors.list.gz) from [IMDB's public datasets](http://www.imdb.com/interfaces). Note that you'll only get male names in this list - if you want female names as well, you'll want to grab `actresses.list.gz`.

There's nothing special about *actor* names in particular that I wanted to capture - this was just the easiest way to get a big list of full names.

## First/last names

Check out [this directory](http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/) for deduped lists of first/last names. 

At around 60k tokens, this dataset is relatively small - you'll probably want to do many epochs of training.

## GitHub repositories

I [used Google BigQuery](https://www.githubarchive.org/#bigquery) to grab all distinct repository names (n=3.7m) from GitHub's 2014 archive. This involved puzzling over a lot of help articles and giving Google my credit card information, so to make things easier for future interested parties, I've dumped the dataset into [a GitHub repo](https://github.com/colinmorris/reponames-dataset).

## Board Games

I grabbed a scrape of board game geek data from [here](https://github.com/ThaWeatherman/scrapers/blob/master/boardgamegeek/games.csv). Thanks to /u/thaweatherman for [posting this on /r/datasets](https://www.reddit.com/r/datasets/comments/3lm8p4/boardgamegeek_data/).

There are around 80k games total, which is more than the personal names dataset, but these names are much longer and high-entropy. I didn't have much luck learning a model of this data.

## Other ideas

It's not hard to imagine other domains we could apply this to. For example, the names/titles of...

- books
- movies
- songs
- bands
- prescription drugs

I was able to find large public datasets for some of these domains (e.g. [the Project Gutenberg catalog](http://www.gutenberg.org/wiki/Gutenberg:Offline_Catalogs) for books), but a common problem was that they would often contain names from many different languages mixed together. Which makes the problem harder by making the data distribution more complex and multi-modal. It also makes it harder to qualitatively assess outputs.

