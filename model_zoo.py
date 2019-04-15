import tensorflow as tf
from tfwrapper import layers
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

## ======================================================================
## Unet for brain segmentation
## Different BN layer for each protocol
## ======================================================================
def unet2D_adaptiveBN(images, domain, training, nlabels): 

    with tf.name_scope('segmenter'):
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        conv1_1 = layers.conv2D_layer_abn(images, 'conv1_1', domain=domain, num_filters=32, training=training, padding='SAME')
        conv1_2 = layers.conv2D_layer_abn(conv1_1, 'conv1_2', domain=domain, num_filters=32, training=training, padding='SAME')
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_abn(pool1, 'conv2_1', domain=domain, num_filters=64, training=training, padding='SAME')
        conv2_2 = layers.conv2D_layer_abn(conv2_1, 'conv2_2', domain=domain, num_filters=64, training=training, padding='SAME')
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_abn(pool2, 'conv3_1', domain=domain, num_filters=128, training=training, padding='SAME')
        conv3_2 = layers.conv2D_layer_abn(conv3_1, 'conv3_2', domain=domain, num_filters=128, training=training, padding='SAME')
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_abn(pool3, 'conv4_1', domain=domain, num_filters=256, training=training, padding='SAME')
        conv4_2 = layers.conv2D_layer_abn(conv4_1, 'conv4_2', domain=domain, num_filters=256, training=training, padding='SAME')
    
        # ====================================
        # Upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upconv3 = layers.reshape_like(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
        concat3 = tf.concat([upconv3, conv3_2], axis=3)
        conv5_1 = layers.conv2D_layer_abn(concat3, 'conv5_1', domain=domain, num_filters=128, training=training, padding='SAME')
        conv5_2 = layers.conv2D_layer_abn(conv5_1, 'conv5_2', domain=domain, num_filters=128, training=training, padding='SAME')
    
        # ====================================
        # Upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upconv2 = layers.reshape_like(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
        concat2 = tf.concat([upconv2, conv2_2], axis=3)
        conv6_1 = layers.conv2D_layer_abn(concat2, 'conv6_1', domain=domain, num_filters=64, training=training, padding='SAME')
        conv6_2 = layers.conv2D_layer_abn(conv6_1, 'conv6_2', domain=domain, num_filters=64, training=training, padding='SAME')
    
        # ====================================
        # Upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upconv1 = layers.reshape_like(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')
        concat1 = tf.concat([upconv1, conv1_2], axis=3)
        conv7_1 = layers.conv2D_layer_abn(concat1, 'conv7_1', domain=domain, num_filters=32, training=training, padding='SAME')
        conv7_2 = layers.conv2D_layer_abn(conv7_1, 'conv7_2', domain=domain, num_filters=32, training=training, padding='SAME')
    
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = layers.conv2D_layer(conv7_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, padding='SAME')

    return pred