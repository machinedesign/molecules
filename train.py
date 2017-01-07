from interface import train

def main():
    params = {
        'family': 'autoencoder',
        'model': {
            'name': 'rnn_rnn_autoencoder',
            'params':{
                'encode_nb_hidden_units': [128],
                'latent_nb_hidden_units': [32],
                'latent_activations': ['linear'],
                'decode_nb_hidden_units': [128],
                'rnn_type': 'GRU',
                'output_activation': {'name': 'axis_softmax' , 'params': {'axis': 2}}
             }
        },
        'data': {
            'train': {
                'pipeline':[
                    {"name": "load_hdf5", 
                     "params": {"filename": "./data/zinc12_processed.h5", "cols": ["X"]}},
                ]
            },
            'transformers':[
            ]
        },
        'report':{
            'outdir': 'out',
            'checkpoint': {
                'loss': 'train_categorical_crossentropy',
                'save_best_only': True
            },
            'metrics': ['categorical_crossentropy'],
            'domain_specific': []
        },
        'optim':{
            'algo': {
                'name': 'adam',
                'params': {'lr': 1e-3}
            },
            'lr_schedule':{
                'name': 'constant',
                'params': {}
            },
            'early_stopping':{
                'name': 'none',
                'params': {
                    'patience_loss': 'categorical_crossentropy',
                    'patience': 5
                }
            },
            'max_nb_epochs': 100,
            'batch_size': 128,
            'pred_batch_size': 128,
            'loss': 'categorical_crossentropy',
            'budget_secs': 86400,
            'seed': 42
        },
    }
    train(params)

if __name__ == '__main__':
    main()
