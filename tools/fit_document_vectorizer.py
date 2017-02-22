import os
import pickle

import numpy as np

from machinedesign.transformers import fit_transformers
from molecules.transformers import DocumentVectorizer

from clize import run

def process(*, filename='data/chembl22.npz', batch_size=128, out='transformers.pkl'):
    data = np.load(filename)
    X = data['X']
    batch_size = 128
    length = max(map(len, X))
    print('max length : {}'.format(length))
    doc = DocumentVectorizer(
        length=length,
        begin_character=True,
        end_character=True, 
        pad=True, 
        onehot=True)
    transformers = [
       doc 
    ]
    def generator():
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size]
    fit_transformers(
        transformers,
        generator
    )
    print(doc.words_)
    print(len(doc.words_))
    with open(os.path.join(out), 'wb') as fd:
        pickle.dump(transformers, fd)

if __name__ == '__main__':
    run(process)
