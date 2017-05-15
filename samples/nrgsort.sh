#!/bin/bash

if [ $# -eq 2 ]
then
    shuf $1 | head -n $2 | sort -t '	' -k 2 -n | uniq | cut -d '	' -f 1
else
    sort -t '	' -k 2 -n $1 | uniq | cut -d '	' -f 1
fi
