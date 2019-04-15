import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

## ======================================================================
# max pooling layer
## ======================================================================
def max_pool_layer2d(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    2D max pooling layer with standard 2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
    strides_aug = [1, strides[0], strides[1], 1]

    op = tf.nn.max_pool(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op

## ======================================================================
## ======================================================================
def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 5:  # 3D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], start_crop[3], 0),
                                     (-1, output_size[1], output_size[2], output_size[3], -1))
        elif len(output_size) == 4:  # 2D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], 0),
                                     (-1, output_size[1], output_size[2], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)

## ======================================================================
## ======================================================================    
def pad_to_size(bottom, output_size):

    ''' 
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for 3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        return padded

    elif len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')
    else:
        raise ValueError('Unexpected input size: %d' % input_size)
       
## ======================================================================
# conv layer
## ======================================================================
def conv2D_layer(bottom,
                 name,
                 kernel_size=(3,3),
                 num_filters=32,
                 strides=(1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):

    '''
    Standard 2D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.variable_scope(name):

        weights = get_weight_variable(weight_shape, name='W', type=weight_init, regularize=True)
        op = tf.nn.conv2d(bottom, filter=weights, strides=strides_augm, padding=padding)

        biases = None
        if add_bias:
            biases = get_bias_variable(bias_shape, name='b')
            op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Add Tensorboard summaries
        _add_summaries(op, weights, biases)

        return op

## ======================================================================
# conv layer with adaptive batch normalization: convolution, followed by batch norm, followed by activation
## ======================================================================
def conv2D_layer_abn(bottom,
                     name,
                     domain,
                     training,
                     kernel_size=(3,3),
                     num_filters=32,
                     strides=(1,1),
                     activation=tf.nn.relu,
                     padding="SAME",
                     weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D convolutional layer.
    Separate BN layer for each protocol
    '''

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)
    
    conv_bn = tf.cond(tf.equal(domain, 'D1'),
                      lambda: tf.layers.batch_normalization(inputs = conv, name = name + '_bn_D1', training = training),
                      lambda: tf.cond(tf.equal(domain, 'D2'),
                                      lambda: tf.layers.batch_normalization(inputs = conv, name = name + '_bn_D2', training = training),
                                      lambda: tf.cond(tf.equal(domain, 'D3'),
                                                      lambda: tf.layers.batch_normalization(inputs = conv, name = name + '_bn_D3', training = training),
                                                      lambda: tf.layers.batch_normalization(inputs = conv, name = name + '_bn_D4', training = training))))

    act = activation(conv_bn)

    return act

## ======================================================================
# reshape
## ======================================================================
def reshape_like(target, size, name):
    
    '''
    target: tensor to be reshaped
    size: shape to which the target tensor should be reshaped to
    '''
    
    target_reshaped = tf.image.resize_bilinear(target, size, name=name)    
    return target_reshaped

## ======================================================================
# variable initializer - for weights
## ======================================================================
def get_weight_variable(shape, name=None, type='xavier_uniform', regularize=True, **kwargs):

    initialise_from_constant = False
    if type == 'xavier_normal':
        initial = xavier_initializer(uniform=False, dtype=tf.float32)
    elif type == 'he_normal':
        initial = variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)

    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.get_variable(name, initializer=initial)
        else:
            weight = tf.get_variable(name, shape=shape, initializer=initial)

    if regularize:
        tf.add_to_collection('weight_variables', weight)

    return weight

## ======================================================================
# variable initializer - for biases
## ======================================================================
def get_bias_variable(shape, name=None, init_value=0.0):

    initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

## ======================================================================
## ======================================================================
def _add_summaries(op, weights, biases):

    # Tensorboard variables
    tf.summary.histogram(weights.name[:-2], weights)
    if biases: tf.summary.histogram(biases.name[:-2], biases)
    tf.summary.histogram(op.op.name + '/activations', op)
