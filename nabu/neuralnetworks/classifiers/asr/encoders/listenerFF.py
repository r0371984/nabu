'''@file listenerFF.py
contains the ListenerFF class'''

import tensorflow as tf
import encoder
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks.ops import pyramid_stack


class ListenerFF(encoder.Encoder):
    '''a feedforward listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''ListenerFF constructor

        Args:
            conf: the config file
            '''

        hidden_dim = int(conf['hidden_numunits'])
        out_dim = int(conf['out_numunits'])
        window = int(conf['frame_window'])

        #create the feedforward layers
        self.hidden_layer = layer.LinearExtended(hidden_dim, window)
        self.outlayer = layer.BLSTMLayer(out_dim)

        super(ListenerFF, self).__init__(conf, name)

    def encode(self, inputs, sequence_lengths, is_training=False):
        '''
        get the high level feature representation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        '''

        outputs = inputs
        output_seq_lengths = sequence_lengths

        with tf.variable_scope('inlayer'):
            #apply the linear layer
            outputs = self.hidden_layer(outputs)

            #apply the nonlinearity
            outputs = tf.nn.relu(outputs)

            if float(self.conf['listener_dropout']) < 1 and is_training:
                outputs = tf.nn.dropout(
                    outputs, float(self.conf['listener_dropout']))


        for l in range(int(self.conf['listener_numlayers'])):

            with tf.variable_scope('layer%d' % l):
                #apply the linear layer
                hidden = self.hidden_layer(outputs)

                #apply the nonlinearity
                outputs = (tf.nn.relu(hidden) + outputs)/2

                if float(self.conf['listener_dropout']) < 1 and is_training:
                    outputs = tf.nn.dropout(
                        outputs, float(self.conf['listener_dropout']))


                #apply the pyramid stack
                outputs, output_seq_lengths = pyramid_stack(outputs,
                                                            output_seq_lengths)


        outputs = self.outlayer(outputs, output_seq_lengths)

        if float(self.conf['listener_dropout']) < 1 and is_training:
            outputs = tf.nn.dropout(outputs,
                                    float(self.conf['listener_dropout']))

        return outputs
