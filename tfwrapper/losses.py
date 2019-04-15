import tensorflow as tf
import numpy as np

## ======================================================================
## ======================================================================
def dice_loss(logits, labels, epsilon=1e-10, from_label=0, to_label=-1):
    '''
    Calculate a dice loss defined as `1-foreround_dice`. Default mode assumes that the 0 label
     denotes background and the remaining labels are foreground. 
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :param from_label: First label to evaluate 
    :param to_label: Last label to evaluate
    :return: Dice loss
    '''

    with tf.name_scope('dice_loss'):

        prediction = tf.nn.softmax(logits)
        intersection = tf.multiply(prediction, labels)
        intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

        l = tf.reduce_sum(prediction, axis=[1, 2])
        r = tf.reduce_sum(labels, axis=[1, 2])

        dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)
        loss = 1 - tf.reduce_mean(tf.slice(dices_per_subj, (0, from_label), (-1, to_label)))

    return loss

## ======================================================================
## ======================================================================
def foreground_dice(logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
    '''
    Pseudo-dice calculated from all voxels (from all subjects) and all non-background labels
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: scalar Dice
    '''

    struct_dice = per_structure_dice(logits, labels, epsilon)
    foreground_dice = tf.slice(struct_dice, (0, from_label),(-1, to_label))

    return tf.reduce_mean(foreground_dice)

## ======================================================================
## ======================================================================
def per_structure_dice(logits, labels, epsilon=1e-10):
    '''
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1])
    '''

    ndims = logits.get_shape().ndims

    prediction = tf.nn.softmax(logits)
    hard_pred = tf.one_hot(tf.argmax(prediction, axis=-1), depth=tf.shape(prediction)[-1])
    
    intersection = tf.multiply(hard_pred, labels)

    if ndims == 5:
        reduction_axes = [1,2,3]
    else:
        reduction_axes = [1,2]

    intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=reduction_axes)  # was [1,2]
    
    l = tf.reduce_sum(hard_pred, axis=reduction_axes)
    r = tf.reduce_sum(labels, axis=reduction_axes)

    dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

    return dices_per_subj

## ======================================================================
## ======================================================================
def pixel_wise_cross_entropy_loss(logits, labels):
    '''
    Simple wrapper for the normal tensorflow cross entropy loss 
    '''

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss