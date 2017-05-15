import fileinput
from collections import Counter

c = Counter()
for line in fileinput.input():
    for char in line:
        if ord('a') <= ord(char) <= ord('z') or ord('A') <= ord(char) <= ord('Z') or char in ' \n':
            continue
        c[char] += 1

for char, count in c.iteritems():
    print '{}\t{}\t{}'.format(count, char, ord(char))
