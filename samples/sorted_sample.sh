#!/bin/bash

sort $1 | uniq | shuf | tail -n 300 | sort -t '	' -k 2 -n > .sortsam.tmp
head -n 100 .sortsam.tmp > cool.tmp
tail -n 100 .sortsam.tmp > hot.tmp
head -n 200 .sortsam.tmp | tail -n 100 > warm.tmp
rm .sortsam.tmp
