import model_zoo
import tensorflow as tf

# ======================================================================
# Model settings
# ======================================================================
model_handle = model_zoo.unet2D_adaptiveBN

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (None, None)
nlabels = 15

# ======================================================================
# training settings
# ======================================================================
training_type = 'Initial' # 'Initial' or 'New'
if training_type is 'Initial':
    max_epochs = 10000
    experiment_name = 'Initial_training' 
if training_type is 'New':
    max_epochs = 5000
    experiment_name = 'New_training' 
    Dnew = 'D1'  # set the closest from the already known domains
batch_size = 16
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
loss_type = 'dice'  # crossentropy/dice
summary_writing_frequency = 20
train_eval_frequency = 100
val_eval_frequency = 100
save_frequency = 500
continue_run = False
debug = True

# ======================================================================
# test settings
# ======================================================================
# iteration number to be loaded after training the model (setting this to zero will load the model with the best validation dice score)
load_this_iter = 0
batch_size_test = 1
test_dataset = 'HCP_T1' # 'HCP_T1' or 'HCP_T2' 'CALTECH'
save_qualitative_results = False
if test_dataset is 'HCP_T1':
    image_depth = 311
    save_results_subscript = 'initial_training_D1' # initial_training_D1 / initial_training_D2
if test_dataset is 'HCP_T2':
    image_depth = 311
    save_results_subscript = 'initial_training_D2' # initial_training_D1 / initial_training_D2
if test_dataset is 'CALTECH':
    image_depth = 256
    save_results_subscript = 'new_training_Dnew' # initial_training_D1 / initial_training_D2 / new_training_Dnew
