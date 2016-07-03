from rbm_softmax import CharBernoulliRBMSoftmax
import numpy as np
from sklearn.utils import check_random_state

# Subclassing here is not quite right. Just a little bit of laziness.
class CompositeRBM(CharBernoulliRBMSoftmax):

    def __init__(self, rbms):
        assert all(model.codec.shape() == rbms[0].codec.shape() for model in rbms)
        assert all(model.codec.leftpad == rbms[0].codec.leftpad for model in rbms)
        assert all(model.codec.alphabet == rbms[0].codec.alphabet for model in rbms)
        self.codec = rbms[0].codec
        self.softmax_shape = self.codec.shape()
        self.intercept_visible_ = np.sum([rbm.intercept_visible_ for rbm in rbms], axis=0)
        # We want to concatenate an H_1 x V array with an H_2 x V array etc. to get a H_sum x V array
        self.components_ = np.concatenate([rbm.components_ for rbm in rbms], axis=0)
        self.intercept_hidden_ = np.concatenate([rbm.intercept_hidden_ for rbm in rbms], axis=0)

        self.rng_ = check_random_state(None)

