from keras.layers import Input
from keras.models import Model
from keras.layers import Flatten

from machinedesign.common import rnn_stack
from machinedesign.common import fully_connected_layers
from machinedesign.common import conv1d_layers

from machinedesign.layers import CategoricalMasking

def rnn_predictor(params, input_shape, output_shape):
    assert len(input_shape) == 2
    assert len(output_shape) == 1
    
    rnn_nb_hidden_units = params['rnn_nb_hidden_units']
    fully_connected_nb_hidden_units = params['fully_connected_nb_hidden_units']
    fully_connected_activations = params['fully_connected_activations'] 
    rnn_type = params['rnn_type']
    output_activation = params['output_activation']
    nb_outputs = output_shape[0]

    inp = Input(input_shape)
    x = inp
    x = CategoricalMasking(mask_char=0)(x)
    x = rnn_stack(x, rnn_nb_hidden_units, rnn_type=rnn_type, return_sequences=False) 
    x = fully_connected_layers(x, fully_connected_nb_hidden_units, fully_connected_activations)
    x = fully_connected_layers(x, [nb_outputs], [output_activation])
    out = x
    model = Model(input=inp, output=out)
    return model 

def conv_predictor(params, input_shape, output_shape):
    assert len(input_shape) == 2
    assert len(output_shape) == 1
    conv_nb_filters = params['conv_nb_filters']
    conv_filter_sizes = params['conv_filter_sizes']
    conv_activations = params['conv_activations']
    fully_connected_nb_hidden_units = params['fully_connected_nb_hidden_units']
    fully_connected_activations = params['fully_connected_activations'] 
    output_activation = params['output_activation']
    nb_outputs = output_shape[0]

    inp = Input(input_shape)
    x = inp
    x = conv1d_layers(
        x,
        conv_nb_filters,
        conv_filter_sizes,
        conv_activations)
    x = Flatten()(x)
    x = fully_connected_layers(x, fully_connected_nb_hidden_units, fully_connected_activations)
    x = fully_connected_layers(x, [nb_outputs], [output_activation])
    out = x
    model = Model(input=inp, output=out)
    return model 

builders = {
    'rnn_predictor': rnn_predictor,
    'conv_predictor': conv_predictor
}
