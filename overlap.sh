#!/bin/bash

lines1=`wc -l $1`
lines2=`wc -l $2`

sort $1 > /tmp/lines1.tmp
sort $2 > /tmp/lines2.tmp
overlap=`comm -12 /tmp/lines1.tmp /tmp/lines2.tmp | wc -l`
echo "Wrote unique lines to uniq.tmp"
comm -23 /tmp/lines1.tmp /tmp/lines2.tmp > uniq.tmp
echo "$overlap/$lines1 lines overlapping"
