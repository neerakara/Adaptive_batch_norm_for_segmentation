import nibabel as nib
import numpy as np
import os
import glob

# ===================================================
# ===================================================
def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False

# ===================================================
# ===================================================
def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

# ===================================================
# ===================================================
def save_nii(img_path, data, affine, header=None):
    '''
    Shortcut to save a nifty file
    '''
    if header == None:
        nimg = nib.Nifti1Image(data, affine=affine)
    else:
        nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

# ===================================================
# ===================================================
def normalise_image(image, norm_type):
    '''
    make image zero mean and unit standard deviation
    '''
    if norm_type == 'zero_mean':
        img_o = np.float32(image.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        return np.divide((img_o - m), s)
    elif norm_type == 'div_by_max':
        return np.divide(image, np.percentile(image,98))
    
# ===================================================
# ===================================================
def get_latest_model_checkpoint_path(folder, name):
    '''
    Returns the checkpoint with the highest iteration number with a given name
    :param folder: Folder where the checkpoints are saved
    :param name: Name under which you saved the model
    :return: The path to the checkpoint with the latest iteration
    '''

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file = file.split('/')[-1]
        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))

# ===============================================================
# Group the segmentation classes into the required categories 
# ===============================================================
def group_segmentation_classes(seg_mask):
    
    seg_mask_modified = group_segmentation_classes_15(seg_mask)
    return seg_mask_modified

    
# ===============================================================
# Group the segmentation classes into the required categories (background, white matter, grey matter)
# ===============================================================
def group_segmentation_classes_3(a):
    
    background_ids = [0, 24, 16] # [background, csf, brain stem]
    cerebellum_wm_ids = [7, 46]
    cerebellum_gm_ids = [8, 47]
    cerebral_wm_ids = [2, 41, 251, 252, 253, 254, 255]
    cerebral_gm_ids = np.arange(1000, 3000)
    thalamus_ids = [10, 49]
    hippocampus_ids = [17, 53]
    amygdala_ids = [18, 54]
    ventricle_ids = [4, 43, 14, 15, 72] # lat, 3rd, 4th, 5th
    choroid_plexus_ids = [31, 63]
    caudate_ids = [11, 50]
    putamen_ids = [12, 51]
    pallidum_ids = [13, 52]
    accumbens_ids = [26, 58]
    ventral_DC_ids = [28, 60]
    misc_ids = [5, 44, 30, 62, 77, 80, 85] # inf lat ventricle, right, left vessel, hypointensities, optic-chiasm
    
    a = np.array(a, dtype = 'uint16')
    b = np.zeros((a.shape[0], a.shape[1], a.shape[2]), dtype = 'uint16')

    unique_ids = np.unique(a)    
    # print("Unique labels in the original segmentation mask:", unique_ids)
    
    for i in unique_ids:
        if (i in cerebral_gm_ids): b[a == i] = 2
        elif (i in background_ids): b[a == i] = 0
        elif (i in cerebellum_gm_ids): b[a == i] = 0
        elif (i in cerebellum_wm_ids): b[a == i] = 0
        elif (i in cerebral_wm_ids): b[a == i] = 1
        elif (i in caudate_ids): b[a == i] = 1
        elif (i in putamen_ids): b[a == i] = 1
        elif (i in pallidum_ids): b[a == i] = 1
        elif (i in accumbens_ids): b[a == i] = 1
        elif (i in ventral_DC_ids): b[a == i] = 1
        elif (i in misc_ids): b[a == i] = 1
        elif (i in thalamus_ids): b[a == i] = 1
        elif (i in hippocampus_ids): b[a == i] = 1
        elif (i in amygdala_ids): b[a == i] = 1
        elif (i in ventricle_ids): b[a == i] = 1    
        elif (i in choroid_plexus_ids): b[a == i] = 0
        else: print('unknown id:', i)
        
    print("labels a unique",np.unique(b))
    return b


# ===============================================================
# Group the segmentation classes into the required categories 
# ===============================================================
def group_segmentation_classes_8(a):
    """
    Args:
    label_data : Freesurfer generated Labels Data of a 3D MRI scan.
    Returns:
    relabelled_data
    """
    
    background_ids = [0, 24, 16] # [background, csf, brain stem]
    cerebellum_wm_ids = [7, 46]
    cerebellum_gm_ids = [8, 47]
    cerebral_wm_ids = [2, 41, 251, 252, 253, 254, 255]
    cerebral_gm_ids = np.arange(1000, 3000)
    thalamus_ids = [10, 49]
    hippocampus_ids = [17, 53]
    amygdala_ids = [18, 54]
    ventricle_ids = [4, 43, 14, 15, 72] # lat, 3rd, 4th, 5th
    choroid_plexus_ids = [31, 63]
    caudate_ids = [11, 50]
    putamen_ids = [12, 51]
    pallidum_ids = [13, 52]
    accumbens_ids = [26, 58]
    ventral_DC_ids = [28, 60]
    misc_ids = [5, 44, 30, 62, 77, 80, 85] # inf lat ventricle, right, left vessel, hypointensities, optic-chiasm
    
    a = np.array(a, dtype = 'uint16')
    b = np.zeros((a.shape[0], a.shape[1], a.shape[2]), dtype = 'uint16')

    unique_ids = np.unique(a)    
    # print("Unique labels in the original segmentation mask:", unique_ids)
    
    for i in unique_ids:
        if (i in cerebral_gm_ids): b[a == i] = 3
        elif (i in background_ids): b[a == i] = 0
        elif (i in cerebellum_gm_ids): b[a == i] = 1
        elif (i in cerebellum_wm_ids): b[a == i] = 2
        elif (i in cerebral_wm_ids): b[a == i] = 4
        elif (i in caudate_ids): b[a == i] = 4
        elif (i in putamen_ids): b[a == i] = 4
        elif (i in pallidum_ids): b[a == i] = 4
        elif (i in accumbens_ids): b[a == i] = 4
        elif (i in ventral_DC_ids): b[a == i] = 4
        elif (i in misc_ids): b[a == i] = 4
        elif (i in thalamus_ids): b[a == i] = 5
        elif (i in hippocampus_ids): b[a == i] = 6
        elif (i in amygdala_ids): b[a == i] = 7
        elif (i in ventricle_ids): b[a == i] = 8    
        elif (i in choroid_plexus_ids): b[a == i] = 8    
        else: print('unknown id:', i)
        
    print("labels a unique",np.unique(b))
    return b

# ===============================================================
# Group the segmentation classes into the required categories 
# ===============================================================
def group_segmentation_classes_15(a):
    """
    Args:
    label_data : Freesurfer generated Labels Data of a 3D MRI scan.
    Returns:
    relabelled_data
    """
    
    background_ids = [0] # [background]
    csf_ids = [24] # [csf]
    brainstem_ids = [16] # [brain stem]    
    cerebellum_wm_ids = [7, 46]
    cerebellum_gm_ids = [8, 47]
    cerebral_wm_ids = [2, 41, 251, 252, 253, 254, 255]
    cerebral_gm_ids = np.arange(1000, 3000)
    cerebral_cortex_ids = [3,42]
    thalamus_ids = [10, 49]
    hippocampus_ids = [17, 53]
    amygdala_ids = [18, 54]
    ventricle_ids = [4, 43, 14, 15, 72] # lat, 3rd, 4th, 5th
    choroid_plexus_ids = [31, 63]
    caudate_ids = [11, 50]
    putamen_ids = [12, 51]
    pallidum_ids = [13, 52]
    accumbens_ids = [26, 58]
    ventral_DC_ids = [28, 60]
    misc_ids = [5, 44, 30, 62, 77, 80, 85] # inf lat ventricle, right, left vessel, hypointensities, optic-chiasm
    
    a = np.array(a, dtype = 'uint16')
    b = np.zeros((a.shape[0], a.shape[1], a.shape[2]), dtype = 'uint16')

    unique_ids = np.unique(a)    
    # print("Unique labels in the original segmentation mask:", unique_ids)
    
    for i in unique_ids:
        if (i in cerebral_gm_ids): b[a == i] = 3
        elif (i in cerebral_cortex_ids): b[a == i] = 3
        elif (i in accumbens_ids): b[a == i] = 3
        elif (i in background_ids): b[a == i] = 0
        elif (i in cerebellum_gm_ids): b[a == i] = 1
        elif (i in cerebellum_wm_ids): b[a == i] = 2
        elif (i in cerebral_wm_ids): b[a == i] = 4
        elif (i in misc_ids): b[a == i] = 4
        elif (i in thalamus_ids): b[a == i] = 5
        elif (i in hippocampus_ids): b[a == i] = 6
        elif (i in amygdala_ids): b[a == i] = 7
        elif (i in ventricle_ids): b[a == i] = 8    
        elif (i in choroid_plexus_ids): b[a == i] = 8    
        elif (i in caudate_ids): b[a == i] = 9
        elif (i in putamen_ids): b[a == i] = 10
        elif (i in pallidum_ids): b[a == i] = 11
        elif (i in ventral_DC_ids): b[a == i] = 12
        elif (i in csf_ids): b[a == i] = 13
        elif (i in brainstem_ids): b[a == i] = 14
        else:
            print('unknown id:', i)
            print('num_voxels:', np.shape(np.where(a==i))[1])
        
    print("Unique labels in the modified segmentation mask: ", np.unique(b))
    
    return b

