from sklearn.utils import issparse
import numpy as np


class NonEncodableTextException(Exception):
    
    def __init__(self, reason=None, *args):
        self.reason = reason
        super(self, NonEncodableTextException).__init__(*args)


class ShortTextCodec(object):
    # TODO: problematic if this char appears in the training text
    FILLER = '$' 

    # Backward-compatibility. Was probably a mistake to have FILLER be a class var rather than instance
    @property
    def filler(self):
        if self.__class__.FILLER in self.alphabet:
            return self.__class__.FILLER
        # Old versions of this class used ' ' as filler
        return ' '

    def __init__(self, extra_chars, maxlength, minlength=0, preserve_case=False):
        assert 0 <= minlength <= maxlength
        if self.FILLER not in extra_chars and maxlength != minlength:
            extra_chars = self.FILLER + extra_chars
        self.maxlen = maxlength
        self.minlen = minlength
        self.char_lookup = {}
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

    @property
    def nchars(self):
        return len(self.alphabet)

    def encode(self, s):
        try:
            if len(s) > self.maxlen: 
                raise NonEncodableTextException(reason='toolong')
            elif (hasattr(self, 'minlen') and len(s) < self.minlen):
                raise NonEncodableTextException(reason='tooshort')
            return ([self.char_lookup[c] for c in s] +
                    [self.char_lookup[self.filler] for _ in range(self.maxlen - len(s))])
        except KeyError:
            raise NonEncodableTextException(reason='illegal_char')

    def decode(self, vec, pretty=False):
        if issparse(vec):
            vec = vec.toarray().reshape(-1)
        assert vec.shape == (self.nchars * self.maxlen,)
        chars = []
        for position_index in range(self.maxlen):
            char_index = np.argmax(vec[position_index * self.nchars:(position_index + 1) * self.nchars])
            char = self.alphabet[char_index]
            if pretty and char == self.FILLER:
                char = ' '
            chars.append(char)
        return ''.join(chars)

    def shape(self):
        """The shape of a set of RBM inputs given this codecs configuration."""
        return (self.maxlen, len(self.alphabet))
