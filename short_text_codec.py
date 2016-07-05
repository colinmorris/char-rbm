from sklearn.utils import issparse
import numpy as np
import random


class NonEncodableTextException(Exception):
    
    def __init__(self, reason=None, *args):
        self.reason = reason
        super(NonEncodableTextException, self).__init__(*args)


class ShortTextCodec(object):
    # TODO: problematic if this char appears in the training text
    FILLER = '$' 

    # If a one-hot vector can't be decoded meaningfully, render this char in its place
    MYSTERY = '?'

    # Backward-compatibility. Was probably a mistake to have FILLER be a class var rather than instance
    @property
    def filler(self):
        if self.__class__.FILLER in self.alphabet:
            return self.__class__.FILLER
        # Old versions of this class used ' ' as filler
        return ' '

    def __init__(self, extra_chars, maxlength, minlength=0, preserve_case=False, leftpad=False):
        assert 0 <= minlength <= maxlength
        if self.FILLER not in extra_chars and maxlength != minlength:
            extra_chars = self.FILLER + extra_chars
        self.maxlen = maxlength
        self.minlen = minlength
        self.char_lookup = {}
        self.leftpad_ = leftpad
        self.alphabet = ''
        for i, o in enumerate(range(ord('a'), ord('z') + 1)):
            self.char_lookup[chr(o)] = i
            self.alphabet += chr(o)
        nextidx = len(self.alphabet)
        for i, o in enumerate(range(ord('A'), ord('Z') + 1)):
            if preserve_case:
                self.char_lookup[chr(o)] = nextidx
                nextidx += 1
                self.alphabet += chr(o)
            else:
                self.char_lookup[chr(o)] = i

        offset = len(self.alphabet)
        for i, extra in enumerate(extra_chars):
            self.char_lookup[extra] = i + offset
            self.alphabet += extra

    def debug_description(self):
        return ' '.join('{}={}'.format(attr, repr(getattr(self, attr, None))) for attr in ['maxlen', 'minlen', 'leftpad', 'alphabet', 'nchars'])

    @property
    def leftpad(self):
        return getattr(self, 'leftpad_', False)

    @property
    def nchars(self):
        return len(self.alphabet)

    @property
    def non_special_char_alphabet(self):
        return ''.join(c for c in self.alphabet if (c != ' ' and c != self.FILLER)) 

    def _encode(self, s, padlen):
        if len(s) > padlen:
            raise NonEncodableTextException(reason='toolong')
        padding = [self.char_lookup[self.filler] for _ in range(padlen - len(s))]
        try:
            payload = [self.char_lookup[c] for c in s]
        except KeyError:
            raise NonEncodableTextException(reason='illegal_char')
        if self.leftpad:
            return padding + payload
        else:
            return payload + padding


    def encode(self, s, mutagen=None):
        if len(s) > self.maxlen: 
            raise NonEncodableTextException(reason='toolong')
        elif (hasattr(self, 'minlen') and len(s) < self.minlen):
            raise NonEncodableTextException(reason='tooshort')
        if mutagen:
            s = mutagen(s)
        return self._encode(s, self.maxlen)

    def encode_onehot(self, s):
        indices = self.encode(s)
        return np.eye(self.nchars)[indices].ravel()

    def decode(self, vec, pretty=False, strict=True):
        # TODO: Whether we should use 'strict' mode depends on whether the model
        # we got this vector from does softmax sampling of visibles. Anywhere this
        # is called on fantasy samples, we should use the model to set this param.
        if issparse(vec):
            vec = vec.toarray().reshape(-1)
        assert vec.shape == (self.nchars * self.maxlen,)
        chars = []
        for position_index in range(self.maxlen):
            # Hack - insert a tab between name parts in binomial mode
            if isinstance(self, BinomialShortTextCodec) and pretty and position_index == self.maxlen/2:
                chars.append('\t')
            subarr = vec[position_index * self.nchars:(position_index + 1) * self.nchars]
            if np.count_nonzero(subarr) != 1 and strict:
                char = self.MYSTERY
            else:
                char_index = np.argmax(subarr)
                char = self.alphabet[char_index]
                if pretty and char == self.FILLER:
                    # Hack
                    char = ' ' if isinstance(self, BinomialShortTextCodec) else ''
            chars.append(char)
        return ''.join(chars)

    def shape(self):
        """The shape of a set of RBM inputs given this codecs configuration."""
        return (self.maxlen, len(self.alphabet))

    def mutagen_nudge(self, s):
        # Mutate a single character chosen uniformly at random.
        # If s is shorter than the max length, include an extra virtual character at the end
        i = random.randint(0, min(len(s), self.maxlen-1))
        def roll(forbidden):
            newchar = random.choice(self.alphabet)
            while newchar in forbidden:
                newchar = random.choice(self.alphabet)
            return newchar
                
        if i == len(s):
            return s + roll(self.FILLER + ' ')
        if i == len(s)-1:
            replacement = roll(' ' + s[-1])
            if replacement == self.FILLER:
                return s[:-1]
            return s[:-1] + roll(' ' + s[-1])
        else:
            return s[:i] + roll(s[i] + self.FILLER) + s[i+1:]


    def mutagen_silhouettes(self, s):
        newchars = []
        for char in s:
            if char == ' ':
                newchars.append(char)
            else:
                newchars.append(random.choice(self.non_special_char_alphabet))
        return ''.join(newchars)
        
    def mutagen_noise(self, s):
        return ''.join(random.choice(self.alphabet) for _ in range(self.maxlen))

class BinomialShortTextCodec(ShortTextCodec):
    """Encodes two-part names (e.g. "John Smith"), padding each part separately
    to the same length. (Presumed to  help learning.)
    """

    def __init__(self, *args, **kwargs):
        super(BinomialShortTextCodec, self).__init__(*args, **kwargs)
        self.separator = ','
        # Hack: require maxlen to be even, and give each part of the name
        # an equal share
        assert self.maxlen % 2 == 0, "Maxlen must be even for binomial codec"

    def encode(self, s, mutagen=None):
        """Encode a name assumed to have the format "last, first" (whitespace insensitive)
        """
        namelen = self.maxlen / 2
        if self.separator not in s:
            first = s
            last = ''
        else:
            try:
                last, first = s.split(self.separator)
            except ValueError:
                raise NonEncodableTextException(reason='too many separators')
        last = last.strip()
        first = first.strip()
        if mutagen:
            first = mutagen(first)
            last = mutagen(last)
        return self._encode(first, namelen) + self._encode(last, namelen)

    # We don't really need to override decode(). It should do basically the right
    # thing (modulo some funny spacing)

    # TODO: Probably *do* need to override some or all mutagen methods. Leaving
    # them for now since they're only necessary for evaluation.
