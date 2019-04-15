# Adaptive Batch Normalization for MRI Segmentation

This is a Tensorflow implementation for the experiments described in this article: "A Lifelong Learning Approach to Brain MR Segmentation Across Scanners and Protocols" 
(https://link.springer.com/chapter/10.1007/978-3-030-00928-1_54)

# Requirements

The code has been tested with tensorflow 1.9.0 and python 3.6.6.

# Running the experiments
For training the initial domains, run 'train_initial_domains.py'. This trains the shared convolutional weights as well as the domain-specific batch normalization weights.

For training on a new domain, first run 'evaluate.py' on the training set of the new domain and identify the closest already learned domain.

Then, initialize the batch normalization parameters of the new domain with those of the closest domain and finetune them with the 'train_new_domain.py' file. In this training, the convolutional weights are not updated.
