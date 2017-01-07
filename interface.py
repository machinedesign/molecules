import pickle
import numpy as np

from machinedesign.autoencoder.interface import train as _train
from machinedesign.autoencoder.interface import default_config
from machinedesign.autoencoder.interface import load
from machinedesign.autoencoder.interface import generate
from machinedesign.transformers import transform
from machinedesign.transformers import onehot

from transformers import DocumentVectorizer

def train(params):
    config = default_config
    transformers = config.transformers.copy()
    transformers.update({
        'DocumentVectorizer': DocumentVectorizer
    })
    config = config._replace(transformers=transformers)
    return _train(params, config=config)
