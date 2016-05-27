#!/bin/bash
awk '{ print length($0) }' $1 | sort | uniq -c
