'''@file listenerACNN.py
contains the listenerACNN code'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks import ops
import encoder
import pdb

class ListenerACNN(encoder.Encoder):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, conf, name=None):

        '''Listener constructor

        Args:
            numblocks: the number of blocks, every block consists of dilated layers with a
                        stacking operation at the end
            numlayers: the number of dilated layers in a block
            linear_output: the output dimension of the linear output layer
            fheight: the height of the filter
            fwidth: the width of the filter
            channels_out: the number of output channels
            dropout: the dropout rate
            name: the name of the Listener'''

        #save the parameters
        self.numblocks = int(conf['listener_numblocks'])
        self.numlayers = int(conf['listener_numlayers'])
        self.dropout = float(conf['listener_dropout'])

        #atrous convolutional layer
        self.aconvlayer = layer.AConvLayerTime(int(conf['filter_width']),
            int(conf['filter_height']),int(conf['filter_depth']))
        #linear feedforward output layer
        self.outlayer = layer.Linear(int(conf['outlayer_units']))

        super(ListenerACNN, self).__init__(conf, name)

    def encode(self, inputs, sequence_lengths, is_training=False):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, feature_dim] tensor
            sequence_length: the length of the input sequences (for every batch)
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [bath_size, max_length/numblocks, linear_output]
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

                hidden = self.aconvlayer(hidden,sequence_lengths,1,'convlayer%d' % s)

                outputs = (tf.nn.relu(hidden) + outputs)/2


        for l in range(self.numblocks):

            outputs, sequence_lengths = ops.channel_stack(outputs,sequence_lengths,
                'stack%d' % l)


            #pdb.set_trace()
            with tf.variable_scope('block%d' % (l+1)):

                #the first layer after a stack cannot have a residual connection
                hidden = self.aconvlayer(outputs, sequence_lengths, 1, 'layer0')
                outputs = tf.nn.relu(hidden)

                for s in range(self.numlayers):
                    hidden = self.aconvlayer(outputs,sequence_lengths, 2**(s+1),
                        'layer%d' % (s+1))
                    hidden = tf.nn.relu(hidden)

                    if self.dropout < 1 and is_training:
                        hidden = tf.nn.dropout(hidden, self.dropout)

                    hidden = self.aconvlayer(hidden,sequence_lengths,1,
                        'convlayer%d' % (s+1))

                    outputs = (tf.nn.relu(hidden) + outputs)/2


        outputs = tf.reshape(outputs,[batch_size,int(outputs.get_shape()[1]),-1])
        outputs = self.outlayer(outputs, 'outlayer')

        if self.dropout < 1 and is_training:
            outputs = tf.nn.dropout(outputs, self.dropout)


        return outputs
