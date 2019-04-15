import os
import glob
import numpy as np
import logging
# import image and other utility functions
import utils
import config.system as sys_config
from shutil import copyfile
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# =============================================================================================
# FUNCTION FOR COPYING RELEVANT FILES FROM DIFFERENT SITES TO THE LOCAL DISK
# =============================================================================================
def copy_site_files():
     
    # set site name
    #sitepath = 'caltech/dicom/triotim/mmilham/abide_28730/'
    sitepath = 'stanford/dicom/signa/mmilham/abide_28730/'
    prepoc_folder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hcp_segmentation/data/preproc_data/abide/stanford/'
    sitefolder = os.path.join(sys_config.orig_data_root_abide, sitepath)
    sitesubjects = sorted(glob.glob(sitefolder + '*'))
    
    # set the destination for this site
    for sub_id in range(len(sitesubjects)):
        
        patname = sitesubjects[sub_id][sitesubjects[sub_id].rfind('/')+1:]
        patdir = prepoc_folder + patname + '/'
        if not tf.gfile.Exists(patdir):
            tf.gfile.MakeDirs(patdir)
        
        sesspath = glob.glob(sitesubjects[sub_id] + '/*')        
        sessname = sesspath[0][sesspath[0].rfind('/')+1:]
        
        # copy main image
        copyfile(sesspath[0] + '/mprage_0001/MPRAGE.nii.gz', patdir + 'MPRAGE.nii.gz')
            
        # read the contents inside the top-level subject directory
        filenames = sorted(glob.glob(sesspath[0] + '/mprage_0001/' + sessname + '/mri/*'))
        
        # search for and copy the relevant files
        for name in filenames:        
            # search for the image file
            if name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/aparc+aseg.mgz':
                logging.info('%d done!' %sub_id)
                copyfile(name, patdir + 'aparc+aseg.mgz')
            elif name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/brain.mgz':
                copyfile(name, patdir + 'brain.mgz')
            elif name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/norm.mgz':
                copyfile(name, patdir + 'norm.mgz')
            elif name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/nu.mgz':
                copyfile(name, patdir + 'nu.mgz')
            elif name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/orig.mgz':
                copyfile(name, patdir + 'orig.mgz')
            elif name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/orig_nu.mgz':
                copyfile(name, patdir + 'orig_nu.mgz')
            elif name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/ribbon.mgz':
                copyfile(name, patdir + 'ribbon.mgz')
            elif name == sesspath[0] + '/mprage_0001/' + sessname + '/mri/T1.mgz':
                copyfile(name, patdir + 'T1.mgz')
            
# ===============================================================     
# This function unzips and pre-processes the data if this has not already been done.
# If this already been done, it reads the processed data and returns it.                    
# ===============================================================                         
def load_data(input_folder,
              preproc_folder,
              idx_start,
              idx_end,
              force_overwrite = False):
    
    # create the pre-processing folder, if it does not exist
    utils.makefolder(preproc_folder)    
    
    logging.info('============================================================')
    logging.info('Loading data...')
        
    # make appropriate filenames according to the requested indices of training, validation and test images
    config_details = 'from%dto%d_' % (idx_start, idx_end)
    filepath_images = preproc_folder + config_details + 'images.npy'
    filepath_masks = preproc_folder + config_details + 'annotations15.npy'
    filepath_affine = preproc_folder + config_details + 'affines.npy'
    filepath_patnames = preproc_folder + config_details + 'patnames.npy'
    
    # if the images have not already been extracted, do so
    if not os.path.exists(filepath_images) or force_overwrite:
        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        images, masks, affines, patnames = prepare_data(input_folder,
                                                        preproc_folder,
                                                        idx_start,
                                                        idx_end)
    else:
        logging.info('Already preprocessed this configuration. Loading now...')
        # read from already created npy files
        images = np.load(filepath_images)
        masks = np.load(filepath_masks)
        affines = np.load(filepath_affine)
        patnames = np.load(filepath_patnames)
        
    return images, masks, affines, patnames


# ===============================================================
# Main function that prepares a dataset from the raw challenge data to an hdf5 dataset.
# Extract the required files from their zipped directories
# ===============================================================
def prepare_data(input_folder,
                 preproc_folder,
                 idx_start,
                 idx_end):
    
    images = []
    affines = []
    patnames = []
    masks = []
        
    # read the foldernames
    foldernames = sorted(glob.glob(input_folder + '*/'))
    logging.info('Number of images in the dataset: %s' % str(len(foldernames)))
        
    # iterate through all indices
    for idx in range(len(foldernames)):
        
        # only consider images within the indices requested
        if (idx < idx_start) or (idx >= idx_end):
            logging.info('skipping subject: %d' %idx)
            continue
        
        logging.info('==============================================')
        
        # get the file name for this subject
        foldername = foldernames[idx]
        
        # read the contents inside the top-level subject directory
        filenames = sorted(glob.glob(foldername + '*'))
        
        # search for the relevant files
        for name in filenames:
            
            # search for the image file
            # if name == foldername + '/norm.mgz': # read the bias field corrected image
            if name == foldername + 'MPRAGE.nii': # read the original image
                # extract the patient name
                _patname = foldername[foldername[:-1].rfind('/') + 1 : -1]
                if _patname == 'A00033264': # this subject has images of a different size
                    continue
                # read the image
                logging.info('reading image: %s' % name)
                _img_data, _img_affine, _img_header = utils.load_nii(name)
                # normalise the image
                _img_data = utils.normalise_image(_img_data, norm_type='div_by_max')
                # save the pre-processed image
                utils.makefolder(preproc_folder + _patname)
                savepath = preproc_folder + _patname + '/preprocessed_image.nii'
                utils.save_nii(savepath, _img_data, _img_affine)
                # append to the list of all images, affines and patient names
                images.append(_img_data)
                affines.append(_img_affine)
                patnames.append(_patname)

            # search for the segmentation file
            # if name == foldername + '/aparc+aseg.mgz': # segmentation mask with ~100 classes
            if name == foldername + 'orig_labels_aligned_with_true_image.nii.gz': # segmentation mask with ~100 classes
                logging.info('reading segmentation mask: %s' % name)
                # extract the patient name
                _patname = foldername[foldername[:-1].rfind('/') + 1 : -1]
                if _patname == 'A00033264': # this subject has images of a different size
                    continue
                # read the segmentation mask
                _seg_data, _seg_affine, _seg_header = utils.load_nii(name)
                # group the segmentation classes as required
                _seg_data = utils.group_segmentation_classes(_seg_data)
                # save the pre-processed segmentation ground truth
                utils.makefolder(preproc_folder + _patname)
                savepath = preproc_folder + _patname + '/preprocessed_gt15.nii'
                utils.save_nii(savepath, _seg_data, _seg_affine)
                # append to the list of all masks
                masks.append(_seg_data)
                print(_seg_data.shape)

    # convert the lists to arrays
    images = np.array(images)
    affines = np.array(affines)
    patnames = np.array(patnames)
    masks = np.array(masks, dtype = 'uint8')
        
    # merge along the y-zis to get a stack of x-z slices, for the images as well as the masks
    images = images.swapaxes(1,2)
    images = images.reshape(-1,images.shape[2], images.shape[3])
    masks = masks.swapaxes(1,2)
    masks = masks.reshape(-1,masks.shape[2], masks.shape[3])
    
    # save the processed images and masks so that they can be directly read the next time
    # make appropriate filenames according to the requested indices of training, validation and test images
    logging.info('Saving pre-processed files...')
    config_details = 'from%dto%d_' % (idx_start, idx_end)
    
    filepath_images = preproc_folder + config_details + 'images.npy'
    filepath_masks = preproc_folder + config_details + 'annotations15.npy'
    filepath_affine = preproc_folder + config_details + 'affines.npy'
    filepath_patnames = preproc_folder + config_details + 'patnames.npy'
    
    np.save(filepath_images, images)
    np.save(filepath_masks, masks)
    np.save(filepath_affine, affines)
    np.save(filepath_patnames, patnames)
                      
    return images, masks, affines, patnames


## ===============================================================
## Main function that runs if this file is run directly
## ===============================================================
if __name__ == '__main__':

    #copy_site_files()

    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'caltech/', sys_config.preproc_folder_abide + 'caltech/', 10, 12, force_overwrite = True) # train    
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'caltech/', sys_config.preproc_folder_abide + 'caltech/', 0, 2, force_overwrite = True) # validation
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'caltech/', sys_config.preproc_folder_abide + 'caltech/', 18, 36, force_overwrite = True) # test
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'caltech/', sys_config.preproc_folder_abide + 'caltech/', 2, 18, force_overwrite = True) # train from scratch   
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    
    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'stanford/', sys_config.preproc_folder_abide + 'stanford/', 11, 13, force_overwrite = True) # train    
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'stanford/', sys_config.preproc_folder_abide + 'stanford/', 1, 3, force_overwrite = True) # validation
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'stanford/', sys_config.preproc_folder_abide + 'stanford/', 20, 39, force_overwrite = True) # test
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    i, g, _, _ = load_data(sys_config.preproc_folder_abide + 'stanford/', sys_config.preproc_folder_abide + 'stanford/', 3, 20, force_overwrite = True) # train from scratch   
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    
    
# ===============================================================
# End of file
# ===============================================================
