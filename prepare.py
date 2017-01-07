import h5py
import json
import pandas as pd
from clize import run
import pickle

from machinedesign.transformers import onehot
from transformers import DocumentVectorizer

def preprocess(filename,out, *, max_length=120, transformer=None):
    max_length = int(max_length)

    data = pd.read_hdf(filename, 'table')
    data = data['structure']
    data = data.values
    
    if transformer:
        with open(transformer, 'rb') as fd:
            tf = pickle.load(fd)
    else:
        tf = DocumentVectorizer(length=max_length, begin_letter=True, pad=True)
        tf.fit(data)
        transformer = out.split('.')[0] + '_transformer.pkl'
        with open(transformer, 'wb') as fd:
            pickle.dump(tf, fd)
    
    hf = h5py.File(out, 'w')

    shape = (len(data), max_length, tf._nb_words)
    dset = hf.create_dataset('X', shape)
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        d = data[i:i+chunk_size]
        d = tf.transform(d)
        d = onehot(d, tf._nb_words)
        dset[i:i+chunk_size] = d
    hf.close()

if __name__ == '__main__':
    run(preprocess)
