import os
import numpy as np
from collections import deque

from scipy.stats import norm

import keras.backend as K

from fluentopt import Bandit
from fluentopt.bandit import ucb_maximize
from fluentopt.transformers import Wrapper
from fluentopt.utils import RandomForestRegressorWithUncertainty

from machinedesign.common import get_layers
from machinedesign.utils import mkdir_path

from machinedesign.autoencoder.interface import train as _train
from machinedesign.autoencoder.interface import default_config
from machinedesign.autoencoder.interface import load as _load
from machinedesign.autoencoder.interface import custom_objects
from machinedesign.autoencoder.interface import get_method as _get_method

from machinedesign.transformers import onehot
from machinedesign.transformers import transform_one
from machinedesign.transformers import inverse_transform_one

from machinedesign.data import intX

from .transformers import DocumentVectorizer
from .transformers import BEGIN_CHARACTER
from .transformers import ZERO_CHARACTER

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
    model_params = params['model']
    folder = model_params['folder']
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
    with open(os.path.join(save_folder, 'generated.txt'), 'w') as fd:
        for doc in text:
            fd.write(doc + '\n')


def get_method(name):
    if name == "greedy":
        return _greedy
    else:
        return _get_method(name)

def _greedy(params, model):
    nb_samples = params['nb_samples']
    seed = params['seed']
    vectorizer = model.transformers[0]

    def pred_func(x, t):
        x = onehot(x, D=vectorizer.nb_words_)
        y = model.predict(x)
        y = y[:, t, :]
        return y

    text = _generate_text_greedy(
        pred_func,
        vectorizer,
        nb_samples=nb_samples,
        method='proba',
        random_state=seed)
    return text


def _generate_text_greedy(pred_func, vectorizer, nb_samples=1,
                          method='argmax', temperature=1,
                          apply_softmax=False,
                          random_state=None):
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

    random_state : int or None
        random state to use for generation.
        Only have effect if method == 'proba'.

    Returns
    -------

    string numpy array of shape (nb_samples,)

    """
    assert method in ('proba', 'argmax')
    if temperature != 1:
        assert apply_softmax
    rng = np.random.RandomState(random_state)
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


def _bayesopt(params):
    nb_suggestions = params['nb_suggestions']
    method = params['generator']['method']
    generator = load(params['generator']['folder'])
    encoder = load(params['encoder']['folder'])
    input_layer = params['encoder']['input_layer']
    code_layer = params['encoder']['code_layer']
    nb_iter = params['nb_iter']
    objective = params['objective']

    layers = list(get_layers(encoder))
    layers = {layer.name: layer for layer in layers}

    input_layer = layers[input_layer]
    code_layer = layers[code_layer]
    encode = K.function(
        [input_layer.input], 
        [code_layer.output])
    get_str =  {}
    def sampler(rng, buffer=deque(), get_str=get_str):
        if buffer:
            return buffer.pop()
        else:
            text = []
            while len(text) == 0:
                text = _run_method(method, generator)
                text = list(filter(molecule.is_valid, text))
            hbuf = transform_one(text, encoder.transformers)
            hbuf, = encode([hbuf])
            hbuf = hbuf.tolist()
            for t, h in zip(text, hbuf):
                get_str[id(h)] = t
            buffer.extendleft(hbuf)
            return buffer.pop()

    def expected_improvement(model, inputs):
        fmax = expected_improvement.fmax
        fmin = -fmax
        mu, std = model.predict(inputs, return_std=True)
        # eq15 from "Efficient Global Optimization of Expensive Black-Box Functions"
        return (fmin - mu) * norm.cdf((fmin - mu) / std) + std * norm.pdf((fmin - mu) / std)

    expected_improvement.fmax = -np.inf
    
    model = Wrapper(RandomForestRegressorWithUncertainty())
    opt = Bandit(
        sampler=sampler, 
        score=expected_improvement, 
        nb_suggestions=nb_suggestions,
        model=model)
    objective_func = _get_objective_func(objective)

    inputs = []
    outputs = []
    for it in range(nb_iter):
        logger.info('Iteration {}'.format(it + 1))
        h = opt.suggest()
        x = get_str[id(h)]
        y = objective_func(x)
        opt.update(x=h, y=y)
        inputs.append(x)
        outputs.append(y)
        logger.info('input : {}, score : {}'.format(x, y))
        expected_improvement.fmax = max(y, expected_improvement.fmax)
    return inputs, outputs 

def _get_objective_func(name):
    return molecule.logp
