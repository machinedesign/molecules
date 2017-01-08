from clize import run

import numpy as np

from interface import train
from interface import generate

import molecule

def train_model():
    params = {
        'family': 'autoencoder',
        'input_col': 'X',
        'output_col': 'X',
        'model': {
            'name': 'rnn',
            'params':{
                'nb_hidden_units': [512],
                'rnn_type': 'GRU',
                'output_activation': {'name': 'axis_softmax' , 'params': {'axis': 'time_features'}}
             }
        },
        'data': {
            'train': {
                'pipeline':[
                    {"name": "load_numpy", 
                     "params": {"filename": "./data/zinc12.npz", "cols": ["X"], "nb": 100000}},
                ]
            },
            'transformers':[
                {'name': 'DocumentVectorizer', 'params': {'length': 120, 'onehot': True}}
            ]
        },
        'report':{
            'outdir': 'out',
            'checkpoint': {
                'loss': 'train_shifted_categorical_crossentropy',
                'save_best_only': True
            },
            'metrics': ['shifted_categorical_crossentropy'],
        },
        'optim':{
            'algo': {
                'name': 'adam',
                'params': {'lr': 1e-4}
            },
            'lr_schedule':{
                'name': 'constant',
                'params': {}
            },
            'early_stopping':{
                'name': 'none',
                'params': {
                    'patience_loss': 'shifted_categorical_crossentropy',
                    'patience': 5
                }
            },
            'max_nb_epochs': 100,
            'batch_size': 128,
            'pred_batch_size': 128,
            'loss': 'shifted_categorical_crossentropy',
            'budget_secs': 86400,
            'seed': 42
        },
    }
    train(params)

def gen():

    data = np.load('data/zinc12.npz')
    X = data['X'][0:100000]
    X = set(X)

    params = {
        'model':{
            'folder': 'out'
        },
        'method':{
            'name': 'greedy',
            'params': {
                'nb_samples': 1000,
                'max_length': 120 
            },
            'save_folder': 'out/gen',
        }
    }
    generate(params)
    i = 0
    for doc in open('out/gen/generated.txt').readlines():
        s = doc[0:-1]
        # if molecule is valid and not in training set
        if molecule.is_valid(s) and s not in X:
            print(s)
            molecule.draw_image(s, 'out/gen/{:05d}.png'.format(i))
            i += 1
if __name__ == '__main__':
    run(train_model, gen)
