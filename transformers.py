from six.moves import map
import numpy as np

ZERO_CHARACTER = 0
BEGIN_CHARACTER = 1

class DocumentVectorizer(object):

    def __init__(self, length=None, begin_letter=True, pad=True):
        self.length = length
        self.begin_letter = begin_letter
        self.pad = pad

    def fit(self, docs):
        all_words = set(word for doc in docs for word in doc)
        all_words = set(all_words)
        all_words.add(ZERO_CHARACTER)
        all_words.add(BEGIN_CHARACTER)
        self._nb_words = len(all_words)
        self._word2int = {w: i for i, w in enumerate(all_words)}
        self._int2word = {i: w for i, w in enumerate(all_words)}
        return self

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def _doc_transform(self, doc):
        doc = list(map(self._word_transform, doc))
        if self.length:
            if len(doc) >= self.length:
                return doc[0:self.length]
            else:
                doc_new = []
                if self.begin_letter:
                    doc_new.append(self._word_transform(BEGIN_CHARACTER))
                doc_new.extend(doc)
                if self.pad:
                    remaining = self.length - len(doc_new)
                    doc_new.extend(list(map(self._word_transform, [ZERO_CHARACTER] * remaining)))
                return doc_new
        else:
            return doc

    def _word_transform(self, word):
        return self._word2int[word]

    def transform(self, docs):
       docs = list(map(self._doc_transform, docs))
       if self.length:
           docs = np.array(docs)
       return docs

    def inverse_transform(self, X):
        docs = []
        for s in X:
            docs.append([self._int2word[w] for w in s])
        return docs
