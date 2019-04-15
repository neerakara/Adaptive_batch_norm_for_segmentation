import tensorflow as tf
from tfwrapper import losses

# ================================================================
# ================================================================
def inference(images, domain, exp_config, training):
    '''
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module. 
    '''

    return exp_config.model_handle(images, domain, training, nlabels=exp_config.nlabels)

# ================================================================
# ================================================================
def loss(logits, labels, nlabels, loss_type):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'crossentropy'/'dice'/'crossentropy_and_dice'
    :return: The segmentation
    '''

    labels = tf.one_hot(labels, depth=nlabels)

    if loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
    elif loss_type == 'crossentropy_and_dice':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels) + 0.2*losses.dice_loss(logits, labels)
    else:
        raise ValueError('Unknown loss: %s' % loss_type)

    return segmentation_loss

# ================================================================
# ================================================================
def predict(images, protocol, exp_config):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    logits = exp_config.model_handle(images, protocol, training=tf.constant(False, dtype=tf.bool), nlabels=exp_config.nlabels)
    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    return logits, mask, softmax

# ================================================================
# ================================================================
def training_step(loss, var_list, optimizer_handle, learning_rate, **kwargs):
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param learning_rate: Learning rate
    :return: The training operation
    '''

    train_op = optimizer_handle(learning_rate = learning_rate).minimize(loss, var_list=var_list)
    opt_memory_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([train_op, opt_memory_update_ops])

    return train_op

# ================================================================
# ================================================================
def evaluation(logits, labels, images, nlabels, loss_type):
    '''
    A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the 
    current foreground Dice score, and also writes example segmentations and imges to to tensorboard.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input image mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss without weight decay, the foreground dice of a minibatch
    '''

    mask = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
    mask_gt = labels

    tf.summary.image('example_gt1', prepare_tensor_for_summary(mask_gt, mode='mask', idx=10, nlabels=nlabels))
    tf.summary.image('example_pred1', prepare_tensor_for_summary(mask, mode='mask', idx=10, nlabels=nlabels))
    tf.summary.image('example_zimg1', prepare_tensor_for_summary(images, mode='image', idx=10))
    
    tf.summary.image('example_gt2', prepare_tensor_for_summary(mask_gt, mode='mask', idx=11, nlabels=nlabels))
    tf.summary.image('example_pred2', prepare_tensor_for_summary(mask, mode='mask', idx=11, nlabels=nlabels))
    tf.summary.image('example_zimg2', prepare_tensor_for_summary(images, mode='image', idx=11))
    
    tf.summary.image('example_gt3', prepare_tensor_for_summary(mask_gt, mode='mask', idx=12, nlabels=nlabels))
    tf.summary.image('example_pred3', prepare_tensor_for_summary(mask, mode='mask', idx=12, nlabels=nlabels))
    tf.summary.image('example_zimg3', prepare_tensor_for_summary(images, mode='image', idx=12))

    nowd_loss, cdice, _, _ = evaluate_losses(logits, labels, nlabels, loss_type)

    return nowd_loss, cdice

# ================================================================
# ================================================================
def evaluate_losses(logits, labels, nlabels, loss_type):
    '''
    A function to compute various loss measures to compare the predicted and ground truth annotations
    '''
    
    nowd_loss = loss(logits, labels, nlabels=nlabels, loss_type=loss_type)

    cdice_structures = losses.per_structure_dice(logits, tf.one_hot(labels, depth=nlabels))
    cdice_foreground = tf.slice(cdice_structures, (0,1), (-1,-1))
    cdice = tf.reduce_mean(cdice_foreground)
    
    return nowd_loss, cdice, cdice_structures, cdice_foreground

# ================================================================
# ================================================================
def prepare_tensor_for_summary(img, mode, idx=0, nlabels=None):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':

        if img.get_shape().ndims == 3:
            V = tf.slice(img, (idx, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (idx, 0, 0, 10), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':

        if img.get_shape().ndims == 3:
            V = tf.slice(img, (idx, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (idx, 0, 0, 0), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (idx, 0, 0, 10, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else:
        raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]

    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V