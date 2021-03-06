'''@file listenerCNNBLSTM.py
contains the listenerCNNBLSTM code'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks import ops
import encoder

class ListenerCNNBLSTM(encoder.Encoder):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''Listener constructor

        Args:
            numlayers: the number of PBLSTM layers
            numunits: the number of units in each layer
            dropout: the dropout rate'''

        #save the parameters
        self.numlayers = int(conf['listener_numlayers'])
        self.dropout = float(conf['listener_dropout'])
        self.numconvlayers = int(conf['listener_numconvlayers'])

        #create the pblstm layer
        #self.pblstm = layer.PBLSTMLayer(int(conf['listener_numunits']))

        #create the blstm layer
        self.blstm = layer.BLSTMLayer(int(conf['listener_numunits']))

	    #create convolutional layer
        self.convlayer = layer.Conv2dLayerStrided(int(conf['filter_width']),
            int(conf['filter_height']),int(conf['filter_depth']))

        super(ListenerCNNBLSTM, self).__init__(conf, name)

    def encode(self, inputs, sequence_lengths, is_training=False):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode
            reuse: Setting this value to true will cause tensorflow to look
                      for variables with the same name in the graph and reuse
                      these instead of creating new variables.
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        """


        batch_size = int(inputs.get_shape()[0])
        outputs = tf.expand_dims(inputs,3)

        with tf.variable_scope('convlayers'):

            for l in range(self.numconvlayers):
                outputs = self.convlayer(outputs, sequence_lengths, 'convlayer%d' % l)

                outputs = ops.batch_norm(outputs, is_training, 'bn%d' % l)

                outputs = tf.nn.relu(outputs)

                if self.dropout < 1 and is_training:
                    outputs = tf.nn.dropout(outputs, self.dropout)

	    outputs = tf.reshape(outputs,[batch_size,int(outputs.get_shape()[1]),-1])

        with tf.variable_scope('blstmlayers'):

            for l in range(self.numlayers):
                outputs = self.blstm(outputs, sequence_lengths, 'blstmlayer%d' % l)

                outputs = ops.batch_norm(outputs, is_training, 'bn%d' % l)

                if self.dropout < 1 and is_training:
                    outputs = tf.nn.dropout(outputs, self.dropout)

        return outputs
