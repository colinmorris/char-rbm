#!/bin/bash

sort $1 | uniq | cut -d '	' -f 1 | shuf
