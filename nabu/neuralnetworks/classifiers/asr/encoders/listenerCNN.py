'''@file listenerCNN.py
contains the listener code'''

import tensorflow as tf
from nabu.neuralnetworks.classifiers import layer
from nabu.neuralnetworks import ops

class ListenerCNN(object):
    '''a listener object

    transforms input features into a high level representation'''

    def __init__(self, numlayers, numunits, dropout=1, name=None):
        '''Listener constructor

        Args:
            numlayers: 
            numunits: the number of units in each layer
            dropout: the dropout rate'''

        #save the parameters
        self.numlayers = numlayers
        self.numunits = numunits
        self.dropout = dropout
	    
	    #create convolutional layers
        self.convlayer = layer.Conv1dLayer(self.numunits,3,1)
        
        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, sequence_lengths, is_training=False, scope=None):
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
  
        with tf.variable_scope(self.scope):

            outputs = inputs
            for l in range(self.numlayers):
            
                hidden = self.convlayer(outputs,sequence_lengths,
                    is_training,'0layer%d' % l)
                hidden = tf.nn.tanh(hidden)
                
                if self.dropout < 1 and is_training:
                    hidden = tf.nn.dropout(hidden, self.dropout)
                
                for s in range(20):
                    hidden2 = self.convlayer(hidden,sequence_lengths,
                        is_training,'layer%d' % (s+20*l))
                    hidden = tf.nn.tanh(hidden2) + hidden                
                
                outputs, sequence_lengths = ops.pyramid_stack(hidden,sequence_lengths,
                    'stack%d' % l)

            outputs = self.convlayer(outputs,sequence_lengths,
                is_training, 'layer%d' % self.numlayers*20)
            outputs = tf.nn.tanh(outputs)

            if self.dropout < 1 and is_training:
                outputs = tf.nn.dropout(outputs, self.dropout)

        self.scope.reuse_variables()

        return outputs
