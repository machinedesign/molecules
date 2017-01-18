import os
import numpy as np
from clize import run

from machinedesign.utils import write_csv

from molecules import molecule
from molecules.interface import train
from molecules.interface import generate

from molecules import transformers

max_length = 73
def train_model():
    params = {
        'family': 'autoencoder',
        'input_col': 'X',
        'output_col': 'X',
        'model': {
            'name': 'rnn',
            'params': {
                'nb_hidden_units': [256, 256],
                'rnn_type': 'LSTM',
                'output_activation': {'name': 'axis_softmax', 'params': {'axis': 'time_features'}},
                'stateful': False
            }
        },
        'data': {
            'train': {
                'pipeline': [
                    {"name": "load_numpy",
                     "params": {"filename": "../data/zinc12.npz",
                                "cols": ["X"],
                                "nb": 10000}},
                ]
            },

            'valid': {
                'pipeline': [
                    {"name": "load_numpy",
                     "params": {"filename": "../data/zinc12.npz",
                                "cols": ["X"],
                                "start": 10000,
                                "nb": 1000}},
                ]
            },
            'transformers': [
                {'name': 'DocumentVectorizer',
                 'params': {
                     'length': max_length,
                     'onehot': True,
                     'begin_character': True,
                     'end_character': True}
                 }
            ]
        },
        'report': {
            'outdir': 'out',
            'checkpoint': {
                'loss': 'train_categorical_crossentropy',
                'save_best_only': True
            },
            'metrics': [
                {'name': 'precision', 'params': {'shifted': True, 'masked': True}},
                {'name': 'categorical_crossentropy', 'params': {'shifted': True, 'masked': True}},
            ],
            'callbacks': [],
        },
        'optim': {
            'algo': {
                'name': 'adam',
                'params': {'lr': 1e-3}
            },
            'lr_schedule': {
                'name': 'decrease_when_stop_improving',
                'params': {
                    'patience': 5,
                    'loss': 'train_categorical_crossentropy',
                    'shrink_factor': 2.
                }
            },
            'early_stopping': {
                'name': 'basic',
                'params': {
                    'patience_loss': 'train_categorical_crossentropy',
                    'patience': 10
                }
            },
            'max_nb_epochs': 1000,
            'batch_size': 128,
            'pred_batch_size': 128,
            'loss': {'name': 'categorical_crossentropy', 'params':{'shifted': True, 'masked': True}},
            'budget_secs': 86400,
            'seed': 42
        },
    }
    train(params)


def gen(*, model_folder='out'):
    folder = os.path.join(model_folder, 'gen/')
    params = {
        'model': {
            'folder': model_folder
        },
        'method': {
            'name': 'greedy',
            'params': {
                'nb_samples': 1000,
                'seed': 42 
            },
            'save_folder': folder,
        }
    }
    generate(params)
    mols = []
    i = 0
    for doc in open('{}/generated.txt'.format(folder)).readlines():
        s = doc[0:-1]
        is_valid = molecule.is_valid(s)
        logp = molecule.logp(s) if is_valid else 'none'
        mols.append({'mol': s, 'is_valid': is_valid, 'logp': logp})
        if is_valid and len(s):
            molecule.draw_image(s, '{}/{:05d}.png'.format(folder, i))
            i += 1
    write_csv(mols, '{}/mols.csv'.format(folder))


if __name__ == '__main__':
    run(train_model, gen)
