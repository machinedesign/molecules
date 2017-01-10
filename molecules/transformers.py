import six
from six.moves import map
import numpy as np

from machinedesign.transformers import onehot

ZERO_CHARACTER = 0
BEGIN_CHARACTER = 1
END_CHARACTER = 2


class DocumentVectorizer(object):
    """
    a document transformer.
    it takes as input a list of documents and returns a vectorized
    representation of the documents.

    documents are either list of str or list of list of str.
    - if the documents is a list of str, then each document is a str so
      the tokens are the individual characters
    - if the documents is a list of list of str, then the tokens are
      the elements of the list (words)

    Parameters
    ----------

    length : int or None
        if int, maximum length of the documents.
        if None, no maximum length is assumed.
        the behavior of length is that if the length of a document
        is greater than length then it is truncated to fit length.
        if pad is True, then all documents will have exactly length size
        (counting the beging, the end character and the zero characters),
        the remaining characters are filled with the zero character.

    begin_character : bool
        whether to add a begin character in the beginning of each document

    end_character : bool
        whether to add an end character in the end of each document

    onehot : bool
        whether to convert the documents into onehot representation when
        calling the method transform.

    """

    def __init__(self, length=None,
                 begin_character=True, end_character=True,
                 pad=True, onehot=False):
        self.length = length
        self.begin_character = begin_character
        self.end_character = end_character
        self.pad = pad
        self.onehot = onehot

        # input_dtype_ is needed by machinedesign.autoencoder iterative_refinement
        # because the input array must be initialized there and the type of the
        # data must be known, by default it is a float, whereas here we have strs.
        if length:
            self.input_dtype_ = '<U{}'.format(length)
        else:
            # if length is not specified, make a 1000 limit to strs
            self.input_dtype_ = '<U1000'
        # input shape is a scalar (str scalar)
        self.input_shape_ = tuple([])
        # to know whether the documents is a list of str (tokens_are_chars is True)
        # or list of list of str (tokens_are_chars is False)
        self._tokens_are_chars = None
        self._ind = 0
        self.words_ = set()
        self.word2int_ = {}
        self.int2word_ = {}
        self.nb_words_ = 0
        self._update(set([ZERO_CHARACTER, BEGIN_CHARACTER, END_CHARACTER]))

    def partial_fit(self, docs):
        # if not set, set _tokens_are_chars
        # TODO in principle I should also verify the coherence
        # of it for all documents
        if self._tokens_are_chars is None:
            if isinstance(docs[0], six.string_types):
                self._tokens_are_chars = True
            else:
                self._tokens_are_chars = False
        words = set(word for doc in docs for word in doc)
        self._update(words)
        return self

    def _update(self, words):
        """this functions adds new words to the vocabulary"""
        new_words = words - self.words_
        for word in new_words:
            self.word2int_[word] = self._ind
            self.int2word_[self._ind] = word
            self._ind += 1
        self.words_ |= new_words
        self.nb_words_ = len(self.words_)

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    def _doc_transform(self, doc):
        doc = list(map(self._word_transform, doc))
        if self.length:
            len_doc = min(len(doc), self.length)  # max possible length is self.length
            if self.begin_character:
                len_doc -= 1
            if self.end_character:
                len_doc -= 1
            doc_new = []
            if self.begin_character:
                doc_new.append(self._word_transform(BEGIN_CHARACTER))
            doc_new.extend(doc[0:len_doc])
            if self.end_character:
                doc_new.append(END_CHARACTER)
            if self.pad:
                remaining = self.length - len(doc_new)
                doc_new.extend(list(map(self._word_transform, [ZERO_CHARACTER] * remaining)))
            return doc_new
        else:
            return doc

    def _word_transform(self, word):
        return self.word2int_[word]

    def transform(self, docs):
        docs = list(map(self._doc_transform, docs))
        if self.length and self.pad:
            # if both length and pad are set, then all documents
            # have the same length, so we can build a numpy array
            # out of docs
            docs = np.array(docs)
        if self.onehot:
            docs = onehot(docs, D=self.nb_words_)
        return docs

    def inverse_transform(self, X):
        if self.onehot:
            X = X.argmax(axis=-1)
        docs = []
        for s in X:
            docs.append([self.int2word_[w] for w in s])
        if self._tokens_are_chars:
            docs = list(map(doc_to_str, docs))
        return docs


def doc_to_str(doc):
    try:
        idx = doc.index(END_CHARACTER)
        doc = doc[0:idx]
    except ValueError:
        pass
    doc = [d for d in doc if d not in (BEGIN_CHARACTER, ZERO_CHARACTER, END_CHARACTER)]
    return ''.join(doc)
