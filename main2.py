'''
自己下载数据集来跑.

'''

import tensorflow as tf
import os

DATA_DIR = os.path.expanduser("d:/data/t2t/data")  # This folder contain the data
TMP_DIR = os.path.expanduser("d:/data/t2t/tmp")
TRAIN_DIR = os.path.expanduser("d:/data/t2t/train")  # This folder contain the model
EXPORT_DIR = os.path.expanduser("d:/data/t2t/export")  # This folder contain the exported model for production
TRANSLATIONS_DIR = os.path.expanduser("d:/data/t2t/translation")  # This folder contain  all translated sequence
EVENT_DIR = os.path.expanduser("d:/data/t2t/event")  # Test the BLEU score
USR_DIR = os.path.expanduser("d:/data/t2t/user")  # This folder contains our data that we want to add

tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(TMP_DIR)
tf.gfile.MakeDirs(TRAIN_DIR)
tf.gfile.MakeDirs(EXPORT_DIR)
tf.gfile.MakeDirs(TRANSLATIONS_DIR)
tf.gfile.MakeDirs(EVENT_DIR)
tf.gfile.MakeDirs(USR_DIR)



PROBLEM = "translate_enfr_wmt32k" # We chose a problem translation English to French with 32.768 vocabulary
MODEL = "transformer" # Our model
HPARAMS = "transformer_big" # Hyperparameters for the model by default
                            # If you have a one gpu, use transformer_big_single_gpu



from tensor2tensor.utils import registry
from tensor2tensor import problems

tmp1=problems.available() #Show all problems
tmp2=registry.list_models() #Show all registered models
print(tmp1)
print(tmp2)


t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR) #下不动,用debug进去看代码里面地址.

