import numpy as np
import keras.backend as K

from machinedesign.objectives import categorical_crossentropy
from machinedesign.metrics import categorical_crossentropy as categorical_crossentropy_metric

def shifted_categorical_crossentropy(y_true, y_pred, masked=True):
    """
    y_pred should be close to y_true shifted to the right by one character,
    metric version. this function computes a masked version if masked is True. 
    the masked version ignores the zero character, that is, it ignores all timesteps
    where the the character to predict is the zero character.
    """
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    if masked:
        return _masked_categorical_crossentropy(y_true, y_pred, backend=K).mean()
    else:
        return categorical_crossentropy(y_true, y_pred)

def shifted_categorical_crossentropy_metric(y_true, y_pred, masked=True):
    """
    metric version of shifted_categorical_crossentropy
    """
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    if masked:
        return _masked_categorical_crossentropy(y_true, y_pred, backend=np)
    else:
        return categorical_crossentropy_metric(y_true, y_pred)

def _masked_categorical_crossentropy(y_true, y_pred, backend=K):
    """
    masked version of categorical crossentropy
    """
    B = backend
    E = y_true.shape[0]#nb_examples
    T = y_true.shape[1]#nb_timesteps
    # masking is done through the zero character
    #WARNING : assumes non zero character has id 0
    # this is enforced by DocumentVectorizer, see
    # the code
    mask = B.not_equal(y_true.argmax(axis=2), 0) 
    # mask shape is (E, T)
    yt = y_true.reshape((-1, y_true.shape[-1])).argmax(axis=1)
    yp = y_pred.reshape((-1, y_pred.shape[-1]))
    L = -B.log(yp[B.arange(yt.shape[0]), yt])
    # L shape is (E * T,)
    L = L.reshape((E, T))
    # mask then divide by the length of the masked sequence insteaf of taking
    # the mean with the length of the padded the sequence.
    L = (L * mask) / mask.sum(axis=1, keepdims=True)
    return L

def precision_metric(y_true, y_pred):
    """
    computes how much the argmax character of y_true and the argmax
    character of y_pred match with each other. the value is  between 0 and 1
    where 1 is the best value, 0 the worst.
    """
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    return (y_true == y_pred)

def shifted_precision_metric(y_true, y_pred, masked=True):
    """
    shifted version of precision_metric
    """
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    if masked:
        return _masked_precision_metric(y_true, y_pred) 
    else:
        return precision_metric(y_true, y_pred)

def _masked_precision_metric(y_true, y_pred):
    """
    masked version of precision metric.
    considers only the characters of y_true that are non zero
    when computing the precision.
    """
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    # masking is done through the zero character
    #WARNING : assumes non zero character has id 0
    # this is enforced by DocumentVectorizer, see
    # the code
    non_zero = (y_true != 0)
    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]
    return (y_true == y_pred)

objectives = {
   'shifted_categorical_crossentropy': shifted_categorical_crossentropy
}

metrics = {
    'shifted_categorical_crossentropy': shifted_categorical_crossentropy_metric,
    'precision' : precision_metric,
    'shifted_precision': shifted_precision_metric,
#TODO remove, these are duplicates of the two previous ones
    'precision_metric' : precision_metric,
    'shifted_precision_metric': shifted_precision_metric
}
