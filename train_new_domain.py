# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import utils
import model as model
import config.system as sys_config
import data_abide

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import unet2D_adaptive_bn as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# Set SGE_GPU environment variable if we are not on the local host
# ==================================================================
sys_config.setup_GPU_environment()

# ==================================================================
# main function for training
# ==================================================================
def run_training():

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)

    # ============================
    # Initialize step number - this is number of mini-batch runs
    # ============================
    init_step = 0

    # ============================
    # Load data
    # ============================
    logging.info('============================================================')
    logging.info('Loading data...')
    
    logging.info('========================================================')
    logging.info('Loading new domain data...')
    logging.info('Reading CALTECH - 1.5T - T1 images...')    
    logging.info('Data root directory: ' + sys_config.orig_data_root_abide)
    imtr, gttr, _, _ = data_abide.load_data(sys_config.preproc_folder_abide + 'caltech/', sys_config.preproc_folder_abide + 'caltech/', 10, 12) # train 
    imvl, gtvl, _, _ = data_abide.load_data(sys_config.preproc_folder_abide + 'caltech/', sys_config.preproc_folder_abide + 'caltech/', 0, 2) # validation
    logging.info('Training Images: %s' %str(imtr.shape))
    logging.info('Validation Images: %s' %str(imvl.shape))
    logging.info('========================================================')
            
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')
        # Placeholders for the images and labels
        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
        mask_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size)
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name = 'labels')
        # Placeholders for the learning rate, protocol (domain) of the input image and to indicate whether the model is being trained or tested
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
        domain_pl = tf.placeholder(tf.string, shape=[], name = 'domain_name')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # Build the graph that computes predictions from the inference model
        # ================================================================
        logits = model.inference(images_pl, domain_pl, exp_config, training = training_pl)
        
        # ================================================================
        # divide the vars into shared, private for the different domains
        # ================================================================
        shared_vars = []
        privateD1_vars = []
        privateD2_vars = []
        privateD3_vars = []
        privateD4_vars = []        
        logging.info('================================')
        logging.info('List of trainable variables in the graph:')
        for v in tf.trainable_variables():
            var_name = v.name
            if 'W' in var_name: shared_vars.append(v)
            elif 'D1' in var_name: privateD1_vars.append(v)
            elif 'D2' in var_name: privateD2_vars.append(v)
            elif 'D3' in var_name: privateD3_vars.append(v)
            elif 'D4' in var_name: privateD4_vars.append(v)
            print(var_name)

        # ================================================================
        # Add ops for calculation of the training loss
        # ================================================================
        loss = model.loss(logits,
                          labels_pl,
                          nlabels=exp_config.nlabels,
                          loss_type=exp_config.loss_type)
        tf.summary.scalar('loss', loss)

        # ================================================================
        # Add optimization ops.
        # Set the vars to be optimized as the BN vars of the closest of the already learned domains
        # ================================================================
        if exp_config.Dnew is 'D1':
            vars_list_to_be_updated = privateD1_vars
        elif exp_config.Dnew is 'D2':
            vars_list_to_be_updated = privateD2_vars
        train_op_bnNew = model.training_step(loss, vars_list_to_be_updated, exp_config.optimizer_handle, learning_rate_pl)

        # ================================================================
        # Add ops for model evaluation
        # ================================================================
        eval_loss = model.evaluation(logits,
                                     labels_pl,
                                     images_pl,
                                     nlabels = exp_config.nlabels,
                                     loss_type = exp_config.loss_type)

        # ================================================================
        # Build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        summary = tf.summary.merge_all()

        # ================================================================
        # Add init ops
        # ================================================================
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        
        # ================================================================
        # Find if any vars are uninitialized
        # ================================================================
        logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()

        # ================================================================
        # create savers for each domain
        # ================================================================
        max_to_keep = 15
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver_best_dice = tf.train.Saver()

        # ================================================================
        # Create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        vl_error_summary = tf.summary.scalar('validation/loss_Dnew', vl_error)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice_Dnew', vl_dice)
        vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss_Dnew', tr_error)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice_Dnew', tr_dice)
        tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])  
        
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        logging.info('============================================================')
        logging.info('initializing all variables...')
        sess.run(init_g, feed_dict = {domain_pl: 'D1'})
        sess.run(init_g, feed_dict = {domain_pl: 'D2'})
        sess.run(init_g, feed_dict = {domain_pl: 'D3'})
        sess.run(init_g, feed_dict = {domain_pl: 'D4'})
        sess.run(init_l)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of uninitialized variables:' )
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        # ================================================================
        # restore shared weights and the shared weights of the closesst domain
        # ================================================================
        logging.info('============================================================')
        model_path = os.path.join(sys_config.log_root, 'initial_training')
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'models/best_dice.ckpt')
        logging.info('Restroring session from: %s' %checkpoint_path)
        saver.restore(sess, checkpoint_path)

        # ================================================================
        # ================================================================        
        step = init_step
        curr_lr = exp_config.learning_rate
        best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================
        for epoch in range(exp_config.max_epochs):

            logging.info('============================================================')
            logging.info('EPOCH %d' % epoch)
        
            # ================================================               
            # alternate between training on different domains                    
            # ================================================
            logging.info('Training on new domain images')
            domain = exp_config.Dnew
            images_tr = imtr
            labels_tr = gttr                
            
            for batch in iterate_minibatches(images_tr, labels_tr, batch_size = exp_config.batch_size):
                
                curr_lr = exp_config.learning_rate
                start_time = time.time()
                x, y = batch

                # ===========================
                # avoid incomplete batches
                # ===========================
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue

                feed_dict = {images_pl: x,
                             labels_pl: y,
                             domain_pl: domain,
                             learning_rate_pl: curr_lr,
                             training_pl: True}
                
                # ===========================
                # update all shared vars, but only one set of batch norm vars
                # ===========================
                _, loss_value = sess.run([train_op_bnNew, loss], feed_dict=feed_dict)

                # ===========================
                # compute the time for this mini-batch computation
                # ===========================
                duration = time.time() - start_time

                # ===========================
                # write the summaries and print an overview fairly often
                # ===========================
                if (step+1) % exp_config.summary_writing_frequency == 0:                    
                    logging.info('Step %d: loss = %.2f (%.3f sec for the last step)' % (step+1, loss_value, duration))
                    
                    # ===========================
                    # print values of domain-specific parameters to ensure that one does not change when the other is being updated
                    # ===========================
                    if exp_config.debug is True:
                        shared_var_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "conv1_1/W:0")[0].eval(session = sess)
                        bn_D1_var_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "conv1_1_bn_D1/beta:0")[0].eval(session = sess)
                        bn_D2_var_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "conv1_1_bn_D2/beta:0")[0].eval(session = sess)
                        bn_D3_var_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "conv1_1_bn_D3/beta:0")[0].eval(session = sess)
                        bn_D4_var_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "conv1_1_bn_D4/beta:0")[0].eval(session = sess)
                        logging.info('value of one of the shared parameters %f' % shared_var_value[0,0,0,0])
                        logging.info('value of one of the D1-specific BN parameters %f' % bn_D1_var_value[0])
                        logging.info('value of one of the D2-specific BN parameters %f' % bn_D2_var_value[0])
                        logging.info('value of one of the D3-specific BN parameters %f' % bn_D3_var_value[0])
                        logging.info('value of one of the D4-specific BN parameters %f' % bn_D4_var_value[0])
                    
                    # ===========================
                    # Update the events file
                    # ===========================
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # ===========================
                # compute the loss on the entire training set
                # ===========================
                if step % exp_config.train_eval_frequency == 0:

                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       domain_pl,
                                                       domain,
                                                       training_pl,
                                                       images_tr,
                                                       labels_tr,
                                                       exp_config.batch_size)                    
                    
                    tr_summary_msg = sess.run(tr_summary, feed_dict={tr_error: train_loss, tr_dice: train_dice})
                    summary_writer.add_summary(tr_summary_msg, step)
                    
                # ===========================
                # Save a checkpoint periodically
                # ===========================
                if step % exp_config.save_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                # ===========================
                # Evaluate the model periodically
                # ===========================
                if step % exp_config.val_eval_frequency == 0:
                    
                    # ===========================
                    # Evaluate against the validation set - all domains
                    # ===========================
                    logging.info('Validation Data Eval1:')
                    [val_loss, val_dice] = do_eval(sess,
                                                   eval_loss,
                                                   images_pl,
                                                   labels_pl,
                                                   domain_pl,
                                                   exp_config.Dnew,
                                                   training_pl,
                                                   imvl,
                                                   gtvl,
                                                   exp_config.batch_size)

                    vl_summary_msg = sess.run(vl_summary, feed_dict={vl_error: val_loss, vl_dice: val_dice})
                    summary_writer.add_summary(vl_summary_msg, step)                    
                    
                    # ===========================
                    # save model if the val dice is the best yet
                    # ===========================
                    if val_dice > best_dice:
                        best_dice = val_dice
                        best_file = os.path.join(log_dir, 'models/best_dice.ckpt')
                        saver_best_dice.save(sess, best_file, global_step=step)
                        logging.info('Found new best dice on validation set! - %f -  Saving model.' % val_dice)

                step += 1
                
        sess.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            domain_placeholder,
            domain,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(images, labels, batch_size=batch_size):

        x, y = batch

        if y.shape[0] < batch_size:
            continue

        feed_dict = {images_placeholder: x,
                     labels_placeholder: y,
                     domain_placeholder: domain,
                     training_time_placeholder: False}

        closs, cdice = sess.run(eval_loss, feed_dict=feed_dict)
        loss_ii += closs
        dice_ii += cdice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice

# ==================================================================
# ==================================================================
def iterate_minibatches(images, labels, batch_size, augment_batch=False):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :param augment_batch: should batch be augmented?
    :return: mini batches
    '''

    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, 100*batch_size, batch_size):

        if b_i + batch_size > n_images:
            continue
        batch_indices = random_indices[b_i:b_i+batch_size]
        X = np.expand_dims(images[batch_indices, ...], axis=-1)
        y = labels[batch_indices, ...]

        yield X, y

# ==================================================================
# ==================================================================
def main():

    # ===========================
    # ===========================
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        tf.gfile.MakeDirs(log_dir + '/models')
    
    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir)

    run_training()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()