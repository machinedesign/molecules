from keras.layers import Layer
from keras import backend as K


class MaskingOneHot(Layer):
    """
    WARNING : This is a copy paste of keras.layers.Masking which assumes that
    the input is a onehot representation of text and rather than specifying
    a mask value, we will give the index of the zero character, which will
    be the character that is used to mask.
    """

    def __init__(self, masking_character=0, **kwargs):
        self.supports_masking = True
        self.masking_character = masking_character
        super(MaskingOneHot, self).__init__(**kwargs)

    def compute_mask(self, x, input_mask=None):
        x = x.argmax(axis=-1)
        mask = K.not_equal(x, self.masking_character)
        return mask

    def call(self, x, mask=None):
        mask = K.not_equal(x, self.masking_character)
        return x * K.cast(mask, K.floatx())

    def get_config(self):
        config = {'masking_character': self.masking_character}
        base_config = super(MaskingOneHot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


layers = {
    'MaskingOneHot': MaskingOneHot
}
