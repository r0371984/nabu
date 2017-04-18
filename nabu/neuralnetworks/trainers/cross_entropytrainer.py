'''@file cross_enthropytrainer.py
contains the CrossEnthropyTrainer'''

import tensorflow as tf
import trainer
from nabu.neuralnetworks import ops

class CrossEntropyTrainer(trainer.Trainer):
    '''A trainer that minimises the cross-enthropy loss, the output sequences
    must be of the same length as the input sequences'''

    def compute_loss(self, targets, logits, logit_seq_length,
                     target_seq_length):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a [batch_size, max_target_length] tensor containing the
                targets
            logits: a [batch_size, max_logit_length, dim] tensor containing the
                logits
            logit_seq_length: the length of all the logit sequences as a
                [batch_size] vector
            target_seq_length: the length of all the target sequences as a
                [batch_size] vector

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_enthropy_loss'):
            output_dim = int(logits.get_shape()[2])

            #put all the tragets on top of each other
            split_targets = tf.unstack(targets)
            for i, target in enumerate(split_targets):
                #only use the real data
                split_targets[i] = target[:target_seq_length[i]]

                #append an end of sequence label
                split_targets[i] = tf.concat(
                    [split_targets[i], [output_dim-1]], 0)

            #concatenate the targets
            nonseq_targets = tf.concat(split_targets, 0)

            #convert the logits to non sequential data
            nonseq_logits = ops.seq2nonseq(logits, logit_seq_length)

            #one hot encode the targets
            #pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets, output_dim)

            '''#collect the attention tensor, with shape [batch_size, sequence_length, output_dim]
            attention = tf.get_collection('attention')
            attention = attention[0]'''

            #compute the cross-enthropy loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=nonseq_logits, labels=nonseq_targets))
                #+ 0.001*tf.reduce_mean(attention_normalize(attention))
                #+ 0.05*tf.reduce_mean(attention_prior(attention))

        return loss

def attention_normalize(attention):
    '''Calculates an extra loss term, it adds a prior for the alignment
        to be normalized in the time direction

    Args:
        attention: the attention tensor of shape
            [batch_size, hl_seq_length, time]

    returns:
        a tensor with a scalar value for every utterance, shape [batch_size]
    '''
    batch_size = int(attention.get_shape()[0])
    hl_length = int(attention.get_shape()[1])

    differences = tf.subtract(tf.reduce_sum(attention,2),tf.ones([batch_size,hl_length]))
    squared = tf.square(differences)

    return tf.reduce_sum(squared, 1)

def attention_prior(attention):
    '''Calculates an extra loss term, it adds a prior for monotonous alignment

    Args:
        attention: the attention tensor of shape
            [batch_size, hl_seq_length, time]

    returns:
        a tensor with a scalar value for every utterance, shape [batch_size]
    '''
    batch_size = int(attention.get_shape()[0])
    dim = int(attention.get_shape()[2])
    #print (batch_size)
    #print (dim)

    '''loss_temp = tf.zeros([batch_size])
    prior_term = loss_temp
    t = 1
    while t <= dim - 1:
        prior_term = tf.add(prior_term,
            tf.maximum(tf.zeros([batch_size]),loss_temp))
        loss_temp = tf.zeros([batch_size])
        i = 1
        print (t)
        while i <= dim - 1:
            loss_temp = tf.add(loss_temp, tf.reduce_sum(tf.subtract(
                tf.slice(attention,[0,0,t],[batch_size,i,1]),
                tf.slice(attention,[0,0,t-1],[batch_size,i,1])),1))
            i = i + 1
        t = t + 1'''


    prior_term = tf.norm(
        tf.subtract(tf.matmul(
        tf.transpose(attention,[0,2,1]), attention),
        tf.eye(num_rows=dim, batch_shape=[batch_size])), axis=[-2,-1])

    return prior_term
