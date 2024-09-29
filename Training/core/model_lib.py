""" This script is for building model graph"""

import tensorflow as tf

from .tools import utils
from .config import logging, ErrorHandler

logger = logging.getLogger('core.ensemble.model_lib')
logger.addHandler(ErrorHandler)


def model_builder(architecture_type='dnn'):
    assert architecture_type in model_name_type_dict, 'models are {}'.format(','.join(model_name_type_dict.keys()))
    return model_name_type_dict[architecture_type]


def _change_scaler_to_list(scaler):
    if not isinstance(scaler, (list, tuple)):
        return [scaler]
    else:
        return scaler


def _dnn_graph(input_dim=None, use_mc_dropout=False):
    """
    The deep neural network based malware detector.
    The implement is based on the paper, entitled ``Adversarial Examples for Malware Detection'',
    which can be found here:  http://patrickmcdaniel.org/pubs/esorics17.pdf

    We slightly change the model architecture by reducing the number of neurons at the last layer to one.
    """
    input_dim = _change_scaler_to_list(input_dim)
    from .model_hp import dnn_hparam
    logger.info(dict(dnn_hparam._asdict()))

    def wrapper(func):
        def graph():
            Dense, _1, _2, _3 = func()
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(input_dim[0],)))
            for units in dnn_hparam.hidden_units:
                model.add(Dense(units, activation=dnn_hparam.activation))
            # model.add(Dense(200, activation=dnn_hparam.activation, name="target_dense_1"))
            # model.add(Dense(200, activation=dnn_hparam.activation, name="target_dense_2"))
            if use_mc_dropout:
                model.add(tf.keras.layers.Dense(dnn_hparam.output_dim, activation=tf.nn.sigmoid))
            else:
                model.add(tf.keras.layers.Dropout(dnn_hparam.dropout_rate))
                model.add(Dense(dnn_hparam.output_dim, activation=tf.nn.sigmoid))
            return model

        return graph

    return wrapper



def _text_cnn_graph(input_dim=None, use_mc_dropout=False):
    """
    deep android malware detection
    The implement is based on the paper, entitled ``Deep Android Malware Detection'',
    which can be found here:  https://dl.acm.org/doi/10.1145/3029806.3029823
    """
    input_dim = _change_scaler_to_list(input_dim)  # dynamical input shape is permitted
    from .model_hp import text_cnn_hparam
    logger.info(dict(text_cnn_hparam._asdict()))

    def wrapper(func):
        def graph():
            Dense, Conv2D, _1, _2 = func()

            class TextCNN(tf.keras.models.Model):
                def __init__(self):
                    super(TextCNN, self).__init__()
                    self.embedding = tf.keras.layers.Embedding(text_cnn_hparam.vocab_size,
                                                               text_cnn_hparam.n_embedding_dim)
                    self.spatial_dropout = tf.keras.layers.SpatialDropout2D(rate=text_cnn_hparam.dropout_rate)
                    self.conv = Conv2D(text_cnn_hparam.n_conv_filters, text_cnn_hparam.kernel_size,
                                       activation=text_cnn_hparam.activation)
                    self.conv_dropout = tf.keras.layers.Dropout(rate=text_cnn_hparam.dropout_rate)
                    self.pooling = tf.keras.layers.GlobalMaxPool2D()  # produce a fixed length vector
                    self.denses = [Dense(neurons, activation='relu') for neurons in text_cnn_hparam.hidden_units]
                    self.dropout = tf.keras.layers.Dropout(text_cnn_hparam.dropout_rate)
                    if use_mc_dropout:
                        self.d_out = tf.keras.layers.Dense(text_cnn_hparam.output_dim, activation=tf.nn.sigmoid)
                    else:
                        self.d_out = Dense(text_cnn_hparam.output_dim, activation=tf.nn.sigmoid)

                def call(self, x, training=False):
                    embed_code = self.embedding(x)
                    # batch_size, seq_length, embedding_dim, 1. Note: seq_length >= conv_kernel_size
                    embed_code = tf.expand_dims(embed_code, axis=-1)
                    if text_cnn_hparam.use_spatial_dropout:
                        embed_code = self.spatial_dropout(embed_code, training=training)

                    conv_x = self.conv(embed_code)
                    if text_cnn_hparam.use_conv_dropout:
                        conv_x = self.conv_dropout(conv_x)

                    flatten_x = self.pooling(conv_x)

                    for i, dense in enumerate(self.denses):
                        flatten_x = dense(flatten_x)
                    if not use_mc_dropout:
                        flatten_x = self.dropout(flatten_x, training=training)
                    return self.d_out(flatten_x)

            return TextCNN()

        return graph

    return wrapper



model_name_type_dict = {
    'dnn': _dnn_graph,
    'text_cnn': _text_cnn_graph,
}

def build_models(input_x, architecture_type, ensemble_type='vanilla', input_dim=None, use_mc_dropout=False):
    builder = model_builder(architecture_type)

    @builder(input_dim, use_mc_dropout)
    def graph():
        return utils.produce_layer(ensemble_type, dropout_rate=0.4)

    model = graph()
    return model(input_x)
