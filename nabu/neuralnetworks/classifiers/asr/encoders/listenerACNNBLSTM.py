'''@file listenerACNNBLSTM.py
contains the listenerACNNBLSTM code'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks import ops
import encoder

class ListenerACNNBLSTM(encoder.Encoder):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):
        '''Listener constructor, there are blocks of dilated CNNs followed by a BLSTM
            output layer

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
        self.aconvlayer = layer.atrous_conv(int(conf['filter_width']),
            int(conf['filter_height']),int(conf['filter_depth']),str(conf['padding']))

        #normal 2d convolution, padding can be added but for now is hardcoded "SAME"
        self.convlayer = layer.Conv2dLayer(int(conf['filter_width']),
            int(conf['filter_height']),int(conf['filter_depth']))

        #create the blstm output layer
        self.outlayer = layer.BLSTMLayer(int(conf['listener_numunits']))

        super(ListenerACNNBLSTM, self).__init__(conf, name)

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
        outputs = tf.expand_dims(inputs,3)

        with tf.variable_scope('block0'):

            for s in range(self.numlayers):
                hidden = self.aconvlayer(outputs,sequence_lengths, 2**s,
                    'layer%d' % (s))
                hidden = tf.nn.relu(hidden)

                if self.dropout < 1 and is_training:
                    hidden = tf.nn.dropout(hidden, self.dropout)

                hidden = self.convlayer(hidden,sequence_lengths,'convlayer%d' % s)

                outputs = (tf.nn.relu(hidden) + outputs)/2


        for l in range(self.numblocks):

            outputs, sequence_lengths = ops.channel_stack(outputs,sequence_lengths,
                'stack%d' % l)


            #pdb.set_trace()
            with tf.variable_scope('block%d' % (l+1)):

                #the first layer after a stack cannot have a residual connection
                hidden = self.convlayer(outputs, sequence_lengths, 'layer0')
                outputs = tf.nn.relu(hidden)

                for s in range(self.numlayers):
                    hidden = self.aconvlayer(outputs,sequence_lengths, 2**(s+1),
                        'layer%d' % (s+1))
                    hidden = tf.nn.relu(hidden)

                    if self.dropout < 1 and is_training:
                        hidden = tf.nn.dropout(hidden, self.dropout)

                    hidden = self.convlayer(hidden,sequence_lengths,
                        'convlayer%d' % (s+1))

                    outputs = (tf.nn.relu(hidden) + outputs)/2

        outputs = tf.reshape(outputs,[batch_size,int(outputs.get_shape()[1]),-1])
        outputs = self.outlayer(outputs, sequence_lengths, 'outlayer')

        if self.dropout < 1 and is_training:
            outputs = tf.nn.dropout(outputs, self.dropout)

        return outputs
