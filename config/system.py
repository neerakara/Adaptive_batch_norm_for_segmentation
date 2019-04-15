import os
import socket
import logging
import subprocess
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY #########################################
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
local_hostnames = ['bmicdl05']

# ==================================================================
# project dirs
# ==================================================================
project_root = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hcp_segmentation/'
bmic_data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/'
project_code_root = os.path.join(project_root, 'code/')
project_data_root = os.path.join(project_root, 'data/')

# ==================================================================
# data dirs
# ==================================================================
orig_data_root_hcp = os.path.join(bmic_data_root,'HCP/3T_Structurals_Preprocessed/')
orig_data_root_abide = os.path.join(bmic_data_root, 'ABIDE/')

# ==================================================================
# dirs where the pre-processed data is stored
# ==================================================================
preproc_folder_hcp = os.path.join(project_data_root,'preproc_data/hcp/')
preproc_folder_abide = os.path.join(project_data_root,'preproc_data/abide/')

# ==================================================================
# log root
# ==================================================================
log_root = os.path.join(project_code_root, 'v0.8/logdir/')

# ==================================================================
# function to set up the GPU environment
# ==================================================================
def setup_GPU_environment():

    hostname = socket.gethostname()
    print('Running on %s' % hostname)
    if not hostname in local_hostnames:
        logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
        if os.environ.get('SGE_GPU') is None:
            gpu_num = subprocess.check_output("grep -h $(whoami) /tmp/lock-gpu*/info.txt | sed  's/^[^0-9]*//;s/[^0-9].*$//'", shell=True).decode('ascii').strip()[0]
        os.environ['SGE_GPU'] = gpu_num
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
        logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])