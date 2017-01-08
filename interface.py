import pickle
import numpy as np

from machinedesign.autoencoder.interface import train as _train
from machinedesign.autoencoder.interface import default_config
from machinedesign.autoencoder.interface import load
from machinedesign.autoencoder.interface import generate
from machinedesign.transformers import transform
from machinedesign.transformers import onehot
from machinedesign.objectives import categorical_crossentropy
from machinedesign.metrics import categorical_crossentropy as categorical_crossentropy_metric

from transformers import DocumentVectorizer

def train(params):
    config = default_config
    transformers = config.transformers.copy()
    transformers['DocumentVectorizer'] = DocumentVectorizer
    objectives = config.objectives.copy()
    objectives['shifted_categorical_crossentropy'] = shifted_categorical_crossentropy
    metrics = config.metrics.copy()
    metrics['shifted_categorical_crossentropy'] = shifted_categorical_crossentropy_metric
    config = config._replace(transformers=transformers, objectives=objectives, metrics=metrics)
    return _train(params, config=config)

def shifted_categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, 0:-1, :]
    y_pred = y_pred[:, 1:, :]
    return categorical_crossentropy(y_true, y_pred)

def shifted_categorical_crossentropy_metric(y_true, y_pred):
    y_true = y_true[:, 0:-1, :]
    y_pred = y_pred[:, 1:, :]
    return categorical_crossentropy_metric(y_true, y_pred)
