import numpy as np
import keras.backend as K

from machinedesign.objectives import categorical_crossentropy
from machinedesign.metrics import categorical_crossentropy as categorical_crossentropy_metric


def shifted_categorical_crossentropy(y_true, y_pred, masked=True):
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    if masked:
        return _masked_categorical_crossentropy(y_true, y_pred, backend=K).mean()
    else:
        return categorical_crossentropy(y_true, y_pred)

def shifted_categorical_crossentropy_metric(y_true, y_pred, masked=True):
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    if masked:
        return _masked_categorical_crossentropy(y_true, y_pred, backend=np)
    else:
        return categorical_crossentropy_metric(y_true, y_pred)

def _masked_categorical_crossentropy(y_true, y_pred, backend=K):
    B = backend
    E = y_true.shape[0]#nb_examples
    T = y_true.shape[1]#nb_timesteps
    # masking is done through the zero character
    zero = 0
    mask = B.not_equal(y_true.argmax(axis=2), zero) 
    # mask shape is (E, T)
    yt = y_true.reshape((-1, y_true.shape[-1])).argmax(axis=1)
    yp = y_pred.reshape((-1, y_pred.shape[-1]))
    L = -B.log(yp[B.arange(yt.shape[0]), yt])
    # L shape is (E * T,)
    L = L.reshape((E, T))
    L = (L * mask) / mask.sum(axis=1, keepdims=True)
    return L

def precision_metric(y_true, y_pred):
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    return (y_true == y_pred).mean()

def shifted_precision_metric(y_true, y_pred, masked=True):
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    return masked_precision_metric(y_true, y_pred) 

def masked_precision_metric(y_true, y_pred):
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    non_zero = (y_true != 0)
    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]
    return (y_true == y_pred)
