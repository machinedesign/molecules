import random
from collections import defaultdict
import numpy as np

class NGram(object):

    def __init__(self, min_gram=1, max_gram=5, begin='^', end='$'):
        self.min_gram = min_gram
        self.max_gram = max_gram
        self.begin = begin
        self.end = end
        self.models_ = {}
        self.vocab_ = []
    
    def fit(self, corpus):
        degs = range(self.min_gram, self.max_gram + 1)
        for deg in degs:
           model =  _build_model(
                corpus, 
                deg=deg,
                begin=self.begin, 
                end=self.end)
           self.models_[deg] = model
        self.vocab_ = list(set(char for doc in corpus for char in doc)) + [self.end]
    
    def generate(self, rng, max_size=15, none_if_doesnt_end=True):
        return _generate(rng, self.models_, begin=self.begin, end=self.end, max_size=max_size, none_if_doesnt_end=none_if_doesnt_end, vocab=self.vocab_)

def _build_model(data, deg=1, begin='^', end='$'):
    freq = defaultdict(_dict_of_float)
    for element in data:
        element = begin + element + end
        for i in range(deg, len(element)):
            freq[element[i - deg:i]][element[i]] += 1
    sum_freqs = {}
    for k, v in freq.items():
        sum_freqs[k] = sum(nb for nb in v.values())
        for kprev in freq[k].keys():
            freq[k][kprev] = float(freq[k][kprev]) / sum_freqs[k]
    return freq

def _dict_of_float():
    return defaultdict(float)

def _generate(rng, models, begin='^', end='$', max_size=15, none_if_doesnt_end=True, vocab=None):
    degs = models.keys()
    degs = sorted(degs, reverse=True)
    maxdeg = max(degs)
    s = begin
    ended = False
    for i in range(1, max_size + 1):
        pr = None
        for deg in degs:
            model = models[deg]
            if i < deg:
                continue
            prev = s[i - deg:i]
            if prev in model:
                pr = model
                break
        if pr is None:
            #if all counts are 0 for all degs, choose the next character randomly
            assert vocab
            char_idx = rng.randint(0, len(vocab) - 1)
            char = vocab[char_idx]
        else:
            chars = list(pr[prev].keys())
            probas = list(pr[prev].values())
            char_idx = np.random.multinomial(1, probas).argmax()
            char = chars[char_idx]
        if char == end:
            ended = True
            break
        s += char
    if none_if_doesnt_end and not ended:
        return None
    else:
        s = s[1:]
        return s
