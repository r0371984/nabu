'''@file listenerBLSTMACNN.py
contains the listenerBLSTMACNN code'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks import ops
import encoder

class ListenerBLSTMACNN(encoder.Encoder):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''Listener constructor, there is an input BLSTM layer followed
        by blocks of dilated CNNs followed by a BLSTM
            output layer, the reduction in time happens in the dilated CNN blocks

        Args:
            numblocks: the number of dilated conv blocks
            numlayers: the number of conv layers in a conv block
            numunits: the number of units in each BLSTM layer
            dropout: the dropout rate
            name: the name of the Listener'''

        #save the parameters
        self.numblocks = int(conf['listener_numblocks'])
        self.numlayers = int(conf['listener_numlayers'])
        self.dropout = float(conf['listener_dropout'])

        #atrous convolutional layer
        self.aconvlayer = layer.AConv2dLayer(int(conf['filter_width']),
            int(conf['filter_height']),int(conf['filter_depth']))

        #create the blstm input and output layer
        self.inlayer = layer.BLSTMLayer(int(conf['listener_numunits']))
        self.outlayer = layer.BLSTMLayer(int(conf['listener_numunits']))

        super(ListenerBLSTMACNN, self).__init__(conf, name)

    def encode(self, inputs, sequence_lengths, is_training=False):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        '''

        batch_size = int(inputs.get_shape()[0])
        outputs = self.inlayer(inputs, sequence_lengths, 'inlayer')
        outputs = tf.expand_dims(outputs,3)

        with tf.variable_scope('block0'):

            for s in range(self.numlayers):
                hidden = self.aconvlayer(outputs,sequence_lengths, 2**s,
                    'layer%d' % (s))

                if self.dropout < 1 and is_training:
                    hidden = tf.nn.dropout(hidden, self.dropout)

                hidden = self.aconvlayer(hidden,sequence_lengths,1,'convlayer%d' % s)

                outputs = (tf.nn.relu(hidden) + outputs)/2


        for l in range(self.numblocks):

            outputs, sequence_lengths = ops.channel_stack(outputs,sequence_lengths,
                'stack%d' % l)


            with tf.variable_scope('block%d' % (l+1)):

		        #the first layer after a stack cannot have a residual connection
                hidden = self.aconvlayer(outputs,sequence_lengths,1,'layer0')
                outputs = tf.nn.relu(hidden)

                for s in range(self.numlayers):
                    hidden = self.aconvlayer(outputs,sequence_lengths, 2**(s+1),
                        'layer%d' % (s+1))

                    if self.dropout < 1 and is_training:
                        hidden = tf.nn.dropout(hidden, self.dropout)

                    hidden = self.aconvlayer(hidden,sequence_lengths,1,
                        'convlayer%d' % (s+1))

                    outputs = (tf.nn.relu(hidden) + outputs)/2


        outputs = tf.reshape(outputs,[batch_size,int(outputs.get_shape()[1]),-1])
        outputs = self.outlayer(outputs, sequence_lengths, 'outlayer')

        if self.dropout < 1 and is_training:
            outputs = tf.nn.dropout(outputs, self.dropout)


        return outputs
