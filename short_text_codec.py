from sklearn.utils import issparse
import numpy as np

class NonEncodableTextException(Exception):
    pass

class ShortTextCodec(object):
    
    FILLER = ' '

    def __init__(self, extra_chars, textlength):
        # Spaces are on the house
        if ' ' not in extra_chars:
            extra_chars = ' ' + extra_chars
        self.maxlen = textlength
        self.char_lookup = {}
        self.alphabet = ''
        for i, o in enumerate(range(ord('a'), ord('z')+1)):
            self.char_lookup[chr(o)] = i
            self.alphabet += chr(o)
        for i, o in enumerate(range(ord('A'), ord('Z')+1)):
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
                raise NonEncodableTextException
            return ([self.char_lookup[c] for c in s] + 
                [self.char_lookup[self.FILLER] for _ in range(self.maxlen - len(s))])
        except KeyError:
            raise NonEncodableTextException
            
    def decode(self, vec):
        if issparse(vec):
            vec = vec.toarray().reshape(-1)
        assert vec.shape == (self.nchars * self.maxlen,)
        chars = []
        for position_index in range(self.maxlen):
            char_index = np.argmax(vec[position_index*self.nchars:(position_index+1)*self.nchars])
            chars.append(self.alphabet[char_index])
        return ''.join(chars)
            
    def shape(self):
        """The shape of a set of RBM inputs given this codecs configuration."""
        return (self.maxlen, len(self.alphabet))
