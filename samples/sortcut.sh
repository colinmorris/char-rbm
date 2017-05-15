#!/bin/bash

sort -t '	' -k 2 -n $1 | cut -d '	' -f 1 
