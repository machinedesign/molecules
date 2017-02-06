import os
import numpy as np
from collections import deque
from functools import partial
from scipy.stats import norm

import keras.backend as K

from machinedesign.common import get_layers
from machinedesign.utils import mkdir_path

from machinedesign.autoencoder.interface import train as _train
from machinedesign.autoencoder.interface import default_config
from machinedesign.autoencoder.interface import load as _load
from machinedesign.autoencoder.interface import custom_objects
from machinedesign.autoencoder.interface import iterative_refinement as _iterative_refinement
from machinedesign.autoencoder.interface import _apply_binarization

from machinedesign.transformers import onehot
from machinedesign.transformers import transform_one
from machinedesign.transformers import inverse_transform_one

from machinedesign.data import intX
from machinedesign.utils import get_axis

from .transformers import DocumentVectorizer
from .transformers import BEGIN_CHARACTER
from .transformers import ZERO_CHARACTER
from .transformers import END_CHARACTER

from .objectives import objectives as custom_objectives
from .objectives import metrics as custom_metrics

from . import molecule

import logging

logging.basicConfig(
    format='%(asctime)s ## %(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y,%I:%M:%S')
logger = logging.getLogger(__name__)


config = default_config
transformers = config.transformers.copy()
transformers['DocumentVectorizer'] = DocumentVectorizer
objectives = config.objectives.copy()
objectives.update(custom_objectives)
custom_objects.update(objectives)
metrics = config.metrics.copy()
metrics.update(custom_metrics)
layers = config.layers.copy()
custom_objects.update(layers)
config = config._replace(
    transformers=transformers,
    objectives=objectives,
    metrics=metrics,
    layers=layers)

def train(params):
    return _train(params, config=config)


def load(folder, custom_objects=custom_objects):
    return _load(folder, custom_objects=custom_objects)


def generate(params):
    method = params['method']
    folder = params['model_folder']
    model = load(folder)
    return _run_method_and_save(method, model)


def _run_method_and_save(method, model):
    text = _run_method(method, model)
    save_folder = method['save_folder']
    _save(text, save_folder)
    return text


def _run_method(method, model):
    name = method['name']
    params = method['params']
    func = get_method(name)
    text = func(params, model)
    return text


def _save(text, save_folder):
    mkdir_path(save_folder)
    np.savez(os.path.join(save_folder,'generated.npz'), X=text)

def get_method(name):
    if name == "greedy":
        return _greedy
    elif name == 'iterative_refinement':
        return partial(_iterative_refinement, apply_binarization=apply_binarization)
    else:
        raise ValueError('Invalid generation procedure : {}'.format(name))

def apply_binarization(name, params, X, rng=np.random):
    if name == 'trim':
        length = params.get('length')
        if length:
            mean_length = length['mean']
            std_length = length['std']
        onehot = params['onehot']
        axis = get_axis(params['axis'])
        if onehot:
            X = (X == X.max(axis=axis, keepdims=True))
        chars = X.argmax(axis=axis)
        for i in range(X.shape[0]):
            char_i = chars[i]
            if (char_i == END_CHARACTER).sum()>0:
                t = (char_i == END_CHARACTER).argmax()
                X[i, t+1:] = 0
                X[i, t+1:, ZERO_CHARACTER]=1
            elif length:
                t = int(rng.normal(mean_length, std_length))
                X[i, t, :] = 0
                X[i, t, END_CHARACTER] = 1
                X[i, t+1:, :] = 0
                X[i, t+1:, ZERO_CHARACTER] = 1
        return X
    else:
        return _apply_binarization(name, params, X, rng=np.random)

def _greedy(params, model):
    nb_samples = params['nb_samples']
    seed = params['seed']
    batch_size = params['batch_size']
    vectorizer = model.transformers[-1]

    def pred_func(x, t):
        x = onehot(x, D=vectorizer.nb_words_)
        y = model.predict(x)
        y = y[:, t, :]
        return y
    
    rng = np.random.RandomState(seed)
    texts = []
    for start in range(0, nb_samples, batch_size):
        end = min(start + batch_size, nb_samples)
        nb = end - start
        text = _generate_text_greedy(
            pred_func,
            vectorizer,
            nb_samples=nb,
            method='proba',
            rng=rng)
        texts.append(text)
    text = np.concatenate(texts, axis=0)
    return text


def _generate_text_greedy(pred_func, vectorizer, nb_samples=1,
                          method='argmax', temperature=1,
                          apply_softmax=False,
                          rng=np.random):
    """
    pred_func : function
        function which predicts the next character based on the first characters.
        It takes two arguments, the current generated text, represented as an int numpy
        array of shape (nb_samples, max_length), and the current timestep t.
        it returns a numpy array of shape (nb_samples, nb_words).
        For each example it returns a score for each word in the vocabulary
        representing the probability distribution of the next word at timestep t+1 for each
        sample.

    vectorizer : DocumentVectorizer
        a DocumentVectorizer instance to convert back to strings

    nb_samples : int
        number of samples to generate (conditioned on the same seed defined by 'cur')

    method : str
        way to generate samples, can be either 'proba' or 'argmax'.
        If 'argmax', the generation is deterministic, each timestep we take the word
        with the maximum probability. Otherwise, if it is 'proba', we take a word
        roportionally to its probability according to the model.

    temperature : float
        temperature used for generation, only have effect if method == 'proba'.
        the predicted scores are multiplied by temperature then softmax is applied
        to the new scores. Big temperature will make the probability distribution
        behave like argmax whereas small temperature will make the probability
        distribution uniform. temperature==1 has no effect,
        In case temperature is used (that is, different than 1), pred_func should return
        the pre-softmax activations of the neural net.

    apply_softmax : bool
        whether to apply softmax to the values returned by pred_func.
        if temperature != 1, then apply_softmax should be True and pred_func must
        return pre-softmax activations.

    rng : np.random.RandomState(defaut=np.random)
        random generator to use.
        Only have effect if method == 'proba'.

    Returns
    -------

    string numpy array of shape (nb_samples,)

    """
    assert method in ('proba', 'argmax')
    if temperature != 1:
        assert apply_softmax
    # initialize the strings with the begin character
    shape = (nb_samples, vectorizer.length)
    gen = np.ones(shape) * ZERO_CHARACTER
    gen[:, 0] = BEGIN_CHARACTER
    gen = intX(gen)
    for i in range(1, vectorizer.length):
        pr = pred_func(gen, i - 1)
        if apply_softmax:
            pr = pr * temperature
            pr = _softmax(pr)
        next_gen = []
        for word_pr in pr:
            if method == 'argmax':
                word_idx = word_pr.argmax()
            elif method == 'proba':
                word_idx = rng.choice(np.arange(len(word_pr)), p=word_pr)
            next_gen.append(word_idx)
        gen[:, i] = next_gen
    gen = onehot(gen, D=vectorizer.nb_words_)
    # WARNING : this assumes that vectorizer have onehot=True
    # it will not work if onehot=Flase in the vectorizer
    gen = vectorizer.inverse_transform(gen)
    return gen

def _softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out
