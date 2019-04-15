import os
import logging
import numpy as np
import tensorflow as tf
import utils
import model as model
from experiments import unet2D_adaptive_bn as exp_config
import data_hcp
import data_abide

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

import sklearn.metrics as met
import config.system as sys_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

sys_config.setup_GPU_environment()

# ============================================================================
# def compute_and_save_results
# ============================================================================
def compute_and_save_results(images, labels, affines, patnames, domain, sess):
    
    logging.info('========================================================')
    
    images_dice = []

    # ================================        
    # go through one subject at a time
    # ================================
    for subject_num in range(images.shape[0] // exp_config.image_depth):

        # ================================
        # print the current subject number to indicate progress
        # ================================
        logging.info('Subject number: %d' % subject_num)
        
        # ================================
        # extract a subject's image
        # ================================
        image = images[subject_num*exp_config.image_depth : (subject_num+1)*exp_config.image_depth,:,:]
        label = labels[subject_num*exp_config.image_depth : (subject_num+1)*exp_config.image_depth,:,:]
        affine = np.squeeze(affines[subject_num:subject_num+1, :, :])
        patname = patnames[subject_num]
        
        # ================================
        # initialize a list for saving the network outputs
        # ================================
        mask_predicted = []
        
        # ================================
        # divide the images into batches
        # ================================
        for b_i in range(0, image.shape[0], batch_size_test):
        
            # ================================            
            # extract the image of this subject and reshape it as required by the network
            # ================================
            X = np.expand_dims(image[b_i:b_i+batch_size_test, ...], axis=-1)
            
            # ================================
            # get the prediction for this batch from the network
            # ================================
            mask_out = sess.run(mask, feed_dict={images_pl: X, domain_pl: domain})
            
            # ================================
            # append the predictions to a list
            # ================================
            mask_predicted.append(mask_out)
            
        # ================================
        # convert to array and merge the first two axes
        # ================================
        mask_predicted = np.array(mask_predicted)
        mask_predicted = mask_predicted.reshape(-1,mask_predicted.shape[2], mask_predicted.shape[3])
        mask_predicted = mask_predicted.astype(float)
        logging.info('shape of predicted output: %s' %str(mask_predicted.shape))
        
        # ================================
        # compute the dice for the subject
        # ================================
        images_dice.append(met.f1_score(label.flatten(), mask_predicted.flatten(), average=None))
        
        # ================================
        # set path for saving qualitative results
        # ================================
        mask_predicted = mask_predicted.swapaxes(1,0)
        if (exp_config.test_dataset is 'HCP_T1') or (exp_config.test_dataset is 'HCP_T2'):
            savepath = sys_config.preproc_folder_hcp + patname + '/' + exp_config.save_results_subscript + '.nii'
        elif exp_config.test_dataset is 'CALTECH':
            savepath = sys_config.preproc_folder_abide + 'caltech/' + patname + '/' + exp_config.save_results_subscript + '.nii'

        # ================================
        # Save
        # ================================
        if exp_config.save_qualitative_results is True:
            utils.save_nii(savepath, mask_predicted, affine)
        
    # ================================
    # print dice statistics
    # ================================
    images_dice = np.array(images_dice)
    for i in range(images_dice.shape[1]):
        dice_stats = compute_stats(images_dice[:,i])
        logging.info('================================================================')
        logging.info('Dice label %d (mean, median, per5, per95) = %.3f, %.3f, %.3f, %.3f' 
                     % (i, dice_stats[0], dice_stats[1], dice_stats[2], dice_stats[3]))
    logging.info('================================================================')
    logging.info('Mean dice over all labels: %.3f' % np.mean(images_dice))
    logging.info('================================================================')
     
# ============================================================================
# ============================================================================
def compute_stats(array):
    
    mean = np.mean(array)
    median = np.median(array)
    per5 = np.percentile(array,5)
    per95 = np.percentile(array,95)
    
    return mean, median, per5, per95

# ============================================================================
# Main function
# ============================================================================
if __name__ == '__main__':

    # ===================================
    # read the test images
    # ===================================
    if exp_config.test_dataset is 'HCP_T1':
        im, gt, af, pn  = data_hcp.load_data(sys_config.orig_data_root_hcp, sys_config.preproc_folder_hcp, 'T1w_', 51, 71)
        test_domain = 'D1'
        path_to_model = os.path.join(sys_config.log_root, 'Initial_training')
    
    elif exp_config.test_dataset is 'HCP_T2':
        im, gt, af, pn  = data_hcp.load_data(sys_config.orig_data_root_hcp, sys_config.preproc_folder_hcp, 'T2w_', 151, 171)
        test_domain = 'D2'
        path_to_model = os.path.join(sys_config.log_root, 'Initial_training')
    
    elif exp_config.test_dataset is 'CALTECH':
        im, gt, af, pn = data_abide.load_data(sys_config.preproc_folder_abide + 'caltech/', sys_config.preproc_folder_abide + 'caltech/', 18, 36)
        test_domain = 'D1'    
        path_to_model = os.path.join(sys_config.log_root, 'New_training')
    
    # ====================================
    # placeholders for images and ground truth labels
    # ====================================    
    nx, ny = exp_config.image_size[:2]
    batch_size_test = exp_config.batch_size_test
    num_channels = exp_config.nlabels
    image_tensor_shape = [batch_size_test] + list(exp_config.image_size) + [1]
    labels_tensor_shape = [None] + list(exp_config.image_size)
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    labels_pl = tf.placeholder(tf.uint8, shape=labels_tensor_shape, name='labels')
    
    # ====================================
    # placeholder for indicating the protocol of the input image
    # ====================================
    domain_pl = tf.placeholder(tf.string, shape=[], name='domain_name')

    # ====================================
    # create predict ops
    # ====================================        
    logits, mask, softmax = model.predict(images_pl, domain_pl, exp_config)
    
    # ====================================
    # saver instance for loading the trained parameters
    # ====================================
    saver = tf.train.Saver()
    
    # ====================================
    # add initializer Ops
    # ====================================
    logging.info('Adding the op to initialize variables...')
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    
    with tf.Session() as sess:
        
        # ====================================
        # Initialize
        # ====================================
        sess.run(init_g, feed_dict = {domain_pl: 'D1'})
        sess.run(init_g, feed_dict = {domain_pl: 'D2'})
        sess.run(init_g, feed_dict = {domain_pl: 'D3'})
        sess.run(init_g, feed_dict = {domain_pl: 'D4'})
        sess.run(init_l)

        # ====================================
        # get the log-directory. the trained models will be saved here.
        # ====================================
        logging.info('========================================================')
        logging.info('Model directory: %s' % path_to_model)
        
        # ====================================
        # load the model
        # ====================================
        if exp_config.load_this_iter == 0:
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'models/best_dice.ckpt')
        else:
            checkpoint_path = os.path.join(path_to_model, 'models/model.ckpt-%d' % exp_config.load_this_iter)
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver.restore(sess, checkpoint_path)
    
        # ====================================
        # evaluate the test images using the graph parameters           
        # ====================================
        compute_and_save_results(im, gt, af, pn, domain=test_domain, sess=sess)        
        logging.info('========================================================')