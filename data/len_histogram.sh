#!/bin/bash
# Print a histogram of line lengths for a text file
awk '{ print length($0) }' "$1" | sort | uniq -c | sort -n -r
