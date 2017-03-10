'''@file listenerCNN.py
contains the listenerCNN code'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks import ops
import encoder

class ListenerCNN(encoder.Encoder):
    '''a listener object, straightforward CNN layers with channel stacking

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''Listener constructor

        Args:
            numblocks: the number of CNN blocks
            numlayers: the number of layers in each block
            dropout: the dropout rate'''

        #save the parameters
        self.numblocks = int(conf['listener_numblocks'])
        self.numlayers = int(conf['listener_numlayers'])
        self.dropout = float(conf['listener_dropout'])

	    #create convolutional layers
        self.convlayer = layer.AConv2dLayer(int(conf['filter_width']),
            int(conf['filter_height']),int(conf['filter_depth']))

        #create ff linear output layer
        self.outlayer = layer.Linear(int(conf['listener_outlayer']))

        super(ListenerCNN, self).__init__(conf, name)

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

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        """

        batch_size = int(inputs.get_shape()[0])
        outputs = tf.expand_dims(inputs,3)

        with tf.variable_scope('block0'):

            for s in range(self.numlayers):
                hidden = self.aconvlayer(outputs,sequence_lengths, 1,
                    'layera%d' % (s))

                if self.dropout < 1 and is_training:
                    hidden = tf.nn.dropout(hidden, self.dropout)

                hidden = self.aconvlayer(hidden,sequence_lengths,1,'layerb%d' % s)

                outputs = (tf.nn.relu(hidden) + outputs)/2


        for l in range(self.numblocks):

            outputs, sequence_lengths = ops.channel_stack(outputs,sequence_lengths,
                'stack%d' % l)


            with tf.variable_scope('block%d' % (l+1)):

		        #the first layer after a stack cannot have a residual connection
                hidden = self.aconvlayer(outputs,sequence_lengths,1,'layer0')
                outputs = tf.nn.relu(hidden)

                for s in range(self.numlayers):
                    hidden = self.aconvlayer(outputs,sequence_lengths, 1,
                        'layera%d' % (s+1))

                    if self.dropout < 1 and is_training:
                        hidden = tf.nn.dropout(hidden, self.dropout)

                    hidden = self.aconvlayer(hidden,sequence_lengths,1,
                        'layerb%d' % (s+1))

                    outputs = (tf.nn.relu(hidden) + outputs)/2

        outputs = tf.reshape(outputs,[batch_size,int(outputs.get_shape()[1]),-1])
        outputs = self.outlayer(outputs, 'outlayer')

        if self.dropout < 1 and is_training:
            outputs = tf.nn.dropout(outputs, self.dropout)


        return outputs
