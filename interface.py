import os
import pickle
import numpy as np

from machinedesign.utils import mkdir_path

from machinedesign.autoencoder.interface import train as _train
from machinedesign.autoencoder.interface import default_config
from machinedesign.autoencoder.interface import load
from machinedesign.autoencoder.interface import custom_objects

from machinedesign.transformers import transform
from machinedesign.transformers import onehot
from machinedesign.data import intX

from transformers import DocumentVectorizer
from transformers import BEGIN_CHARACTER

from objectives import shifted_categorical_crossentropy
from objectives import shifted_categorical_crossentropy_metric

config = default_config
transformers = config.transformers.copy()
transformers['DocumentVectorizer'] = DocumentVectorizer

objectives = config.objectives.copy()
objectives['shifted_categorical_crossentropy'] = shifted_categorical_crossentropy

metrics = config.metrics.copy()
metrics['shifted_categorical_crossentropy'] = shifted_categorical_crossentropy_metric
config = config._replace(transformers=transformers, objectives=objectives, metrics=metrics)

custom_objects.update(objectives)

def train(params):
    return _train(params, config=config)

def generate(params):
    method = params['method']
    model_params = params['model']
    folder = model_params['folder']
    model = load(folder, custom_objects=custom_objects)
    return _run_method(method, model)

def _run_method(method, model):
    name = method['name']
    params = method['params']
    save_folder = method['save_folder']
    func = get_method(name)
    return func(params, model, save_folder)

def get_method(name):
    return {'greedy': _greedy}[name]

def _greedy(params, model, folder):
    nb_samples = params['nb_samples']
    max_length = params['max_length']
    vectorizer = model.transformers[0]

    def pred_func(x):
        x = onehot(x, D=vectorizer.nb_words_)
        y = model.predict(x)
        y = y[:, -1, :]
        return y

    text = _generate_text_greedy(
            pred_func, 
            vectorizer, 
            nb_samples=nb_samples, 
            max_length=max_length,
            method='proba')
    mkdir_path(folder)
    with open(os.path.join(folder, 'generated.txt'), 'w') as fd:
        for doc in text:
            fd.write(doc + '\n')
    return text

def _generate_text_greedy(pred_func, vectorizer, nb_samples=1, max_length=10, 
                          cur=None,  method='argmax', temperature=1, 
                          apply_softmax=False, random_state=None):
    """

    pred_func : function
        function which predicts the next character based on a the first characters.
        It takes (nb_samples, nb_timesteps) as input and returns (nb_samples, nb_words) as output.
        For each example it returns a score for each word in the vocabulary.

    vectorizer : DocumentVectorizer
        a DocumentVectorizer instance to convert back to strings

    cur : str
        cur text to condition on (text seed), otherwise initialized by the begin character.
        the shape of cur should be (nb_samples, nb_initial_timesteps)

    nb_samples : int
        number of samples to generate (conditioned on the same seed defined by 'cur')

    max_length : int
        number of characters to generate
    
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
        return pre-softmx activations.

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
    nb_words = vectorizer.nb_words_
    if cur is None:
        # initialize the 'seed' with random words
        shape = (nb_samples, vectorizer.length + max_length)
        gen = np.ones(shape) * BEGIN_CHARACTER
        start = vectorizer.length
    else:
        # initialize the seed by cur
        assert len(cur) == nb_samples
        gen = np.ones((len(cur), cur.shape[1] + max_length))
        start = cur.shape[1]
        gen[:, 0:start] = cur
    gen = intX(gen)
    for i in range(start, start + max_length):
        pr = pred_func(gen[:, i - start:i])
        if apply_softmax:
            pr = pr * temperature
            pr = softmax(pr)
        next_gen = []
        for word_pr in pr:
            if method == 'argmax':
                word_idx = word_pr.argmax()  # only take argmax
            elif method == 'proba':
                word_idx = rng.choice(np.arange(len(word_pr)), p=word_pr)
            next_gen.append(word_idx)
        gen[:, i] = next_gen
    gen = gen[:, start:]
    gen = onehot(gen, D=vectorizer.nb_words_)
    #WARNING : this assumes that vectorizer have onehot=True
    # it will not work if onehot=Flase
    gen = vectorizer.inverse_transform(gen)
    return gen
