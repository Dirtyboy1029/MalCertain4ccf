from collections import namedtuple

_TRAIN_HP_TEMPLATE = namedtuple('training',
                                ['random_seed', 'n_epochs', 'batch_size', 'learning_rate', 'clipvalue', 'interval'])
train_hparam = _TRAIN_HP_TEMPLATE(random_seed=23456,
                                  n_epochs=30,
                                  batch_size=8,
                                  learning_rate=0.0001,
                                  clipvalue=100.,
                                  interval=2  # saving model weights
                                  )

_MC_DROPOUT_HP_TEMPLATE = namedtuple('mc_dropout', ['dropout_rate', 'n_sampling'])
mc_dropout_hparam = _MC_DROPOUT_HP_TEMPLATE(dropout_rate=0.4,
                                            n_sampling=10
                                            )
_BAYESIAN_HP_TEMPLATE = namedtuple('bayesian', ['n_sampling'])
bayesian_ensemble_hparam = _BAYESIAN_HP_TEMPLATE(n_sampling=10)

_DNN_HP_TEMPLATE = namedtuple('DNN',
                              ['hidden_units', 'dropout_rate', 'activation', 'output_dim'])

dnn_hparam = _DNN_HP_TEMPLATE(hidden_units=[200,200],
                              # DNN has two hidden layers with each having 200 neurons
                              dropout_rate=0.4,
                              activation='relu',
                              output_dim=1  # binary classification#
                              )

_TEXT_CNN_HP_TEMPLATE = namedtuple('textCNN',
                                   ['hidden_units', 'dropout_rate', 'activation', 'output_dim',
                                    'vocab_size', 'n_embedding_dim', 'n_conv_filters', 'kernel_size',
                                    'max_sequence_length',
                                    'use_spatial_dropout', 'use_conv_dropout'])

text_cnn_hparam = _TEXT_CNN_HP_TEMPLATE(hidden_units=[200, 200],
                                        dropout_rate=0.4,
                                        activation='relu',
                                        vocab_size=256,
                                        n_embedding_dim=8,
                                        n_conv_filters=64,
                                        kernel_size=8,
                                        max_sequence_length=500000,  # shall be large for promoting accuracy
                                        use_spatial_dropout=False,
                                        use_conv_dropout=False,
                                        output_dim=1
                                        )

