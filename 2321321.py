#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Styleoshin/tensor2tensor/blob/Transformer_tutorial/tensor2tensor/notebooks/Transformer_translate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Welcome to the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) Colab
#
# Tensor2Tensor, or T2T for short, is a library of deep learning models and datasets designed to make deep learning more accessible and [accelerate ML research](https://research.googleblog.com/2017/06/accelerating-deep-learning-research.html). In this notebook we will see how to use this library for a translation task by exploring the necessary steps. We will see how to define a problem, generate the data, train the model and test the quality of it, and we will translate our sequences and we visualize the attention. We will also see how to download a pre-trained model.

# In[ ]:


#@title
# Copyright 2018 Google LLC.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[ ]:


# Install deps
# get_ipython().system('pip install -q -U tensor2tensor')


# #1. Initialization
#

# ##1.1. Make some directories

# In[ ]:

##
import tensorflow as tf
import os

DATA_DIR = os.path.expanduser("/t2t/data") # This folder contain the data
TMP_DIR = os.path.expanduser("/t2t/tmp")
TRAIN_DIR = os.path.expanduser("/t2t/train") # This folder contain the model
EXPORT_DIR = os.path.expanduser("/t2t/export") # This folder contain the exported model for production
TRANSLATIONS_DIR = os.path.expanduser("/t2t/translation") # This folder contain  all translated sequence
EVENT_DIR = os.path.expanduser("/t2t/event") # Test the BLEU score
USR_DIR = os.path.expanduser("/t2t/user") # This folder contains our data that we want to add

tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(TMP_DIR)
tf.gfile.MakeDirs(TRAIN_DIR)
tf.gfile.MakeDirs(EXPORT_DIR)
tf.gfile.MakeDirs(TRANSLATIONS_DIR)
tf.gfile.MakeDirs(EVENT_DIR)
tf.gfile.MakeDirs(USR_DIR)


# ## 1.2. Init parameters
#
#
#
#

# In[ ]:


PROBLEM = "translate_enfr_wmt32k" # We chose a problem translation English to French with 32.768 vocabulary
MODEL = "transformer" # Our model
HPARAMS = "transformer_big" # Hyperparameters for the model by default
                            # If you have a one gpu, use transformer_big_single_gpu


# In[ ]:


#Show all problems and models

from tensor2tensor.utils import registry
from tensor2tensor import problems

problems.available() #Show all problems
registry.list_models() #Show all registered models
print(problems.available())
#or
##
#Command line
# get_ipython().system('t2t-trainer --registry_help #Show all problems')
# get_ipython().system('t2t-trainer --problems_help #Show all models')


# # 2. Data generation
#
# Generate the data (download the dataset and generate the data).
#
# ---
#
#  You can choose between command line or code.

# ## 2.1. Generate with terminal
# For more information: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_datagen.py

# In[ ]:


# get_ipython().system('t2t-datagen   --data_dir=$DATA_DIR   --tmp_dir=$TMP_DIR   --problem=$PROBLEM   --t2t_usr_dir=$USR_DIR')


# ## 2.2. Generate with code

# In[ ]:

##

#只需要指定PROBLEM就够了.
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR)