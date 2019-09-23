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

os.environ['HOME']='/data'
print(os.environ)#下面路径都加~即可,表示都在/data目录中操作了!!!!!!!!!
DATA_DIR = os.path.expanduser("~/t2t/data") # This folder contain the data
print(DATA_DIR)







DATA_DIR = os.path.expanduser("~/t2t/data") # This folder contain the data
TMP_DIR = os.path.expanduser("~/t2t/tmp")
TRAIN_DIR = os.path.expanduser("~/t2t/train") # This folder contain the model
EXPORT_DIR = os.path.expanduser("~/t2t/export") # This folder contain the exported model for production
TRANSLATIONS_DIR = os.path.expanduser("~/t2t/translation") # This folder contain  all translated sequence
EVENT_DIR = os.path.expanduser("~/t2t/event") # Test the BLEU score
USR_DIR = os.path.expanduser("~/t2t/user") # This folder contains our data that we want to add
 
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


# PROBLEM = "translate_enzh_wmt32k" # We chose a problem translation English to French with 32.768 vocabulary
PROBLEM = "translate_enzh_wmt8k" # We chose a problem translation English to French with 32.768 vocabulary
MODEL = "transformer" # Our model
HPARAMS = "transformer_big_single_gpu" # Hyperparameters for the model by default
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

#只需要指定PROBLEM就够了.
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR) 




#注意python console 调用的不是63342端口.




##



import warnings
warnings.filterwarnings("ignore")
# # 3. Train the model
# 
# 
# 
# 

# ##3.1. Init parameters
# 
# You can choose between command line or code.
# 
# ---
# 
#  batch_size :  a great value of preference.
# 
# ---
# train_steps : research paper mentioned 300k steps with 8 gpu on big transformer. So if you have 1 gpu, you will need to train the model x8 more. (https://arxiv.org/abs/1706.03762 for more information).
# 
# 

# In[ ]:


train_steps = 300000 # Total number of train steps for all Epochs
eval_steps = 1 # Number of steps to perform for each evaluation
batch_size = 100
save_checkpoints_steps = 1000
ALPHA = 0.1
schedule = "continuous_train_and_eval"


from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems

# Init Hparams object from T2T Problem
hparams = create_hparams(HPARAMS)

# Make Changes to Hparams
hparams.batch_size = batch_size
hparams.learning_rate = ALPHA
#hparams.max_length = 256

# Can see all Hparams with code below
#print(json.loads(hparams.to_json())


# create_run_config : https://github.com/tensorflow/tensor2tensor/blob/28adf2690c551ef0f570d41bef2019d9c502ec7e/tensor2tensor/utils/trainer_lib.py#L105
# 
# ---
# 
# 
# create_experiment : https://github.com/tensorflow/tensor2tensor/blob/28adf2690c551ef0f570d41bef2019d9c502ec7e/tensor2tensor/utils/trainer_lib.py#L611

# In[ ]:


RUN_CONFIG = create_run_config(
      model_dir=TRAIN_DIR,
      model_name=MODEL,
      save_checkpoints_steps= save_checkpoints_steps
)

tensorflow_exp_fn = create_experiment(
        run_config=RUN_CONFIG,
        hparams=hparams,
        model_name=MODEL,
        problem_name=PROBLEM,
        data_dir=DATA_DIR, 
        train_steps=train_steps, 
        eval_steps=eval_steps, 
        #use_xla=True # For acceleration
    ) 

tensorflow_exp_fn.train_and_evaluate()
##

# #4. See the BLEU score

# In[ ]:


#INIT FILE FOR TRANSLATE

SOURCE_TEST_TRANSLATE_DIR = TMP_DIR+"/dev/newstest2014-fren-src.en.sgm"
REFERENCE_TEST_TRANSLATE_DIR = TMP_DIR+"/dev/newstest2014-fren-ref.en.sgm"
BEAM_SIZE=1


# In[ ]:


# get_ipython().system('t2t-translate-all   --source=$SOURCE_TEST_TRANSLATE_DIR   --model_dir=$TRAIN_DIR   --translations_dir=$TRANSLATIONS_DIR   --data_dir=$DATA_DIR   --problem=$PROBLEM   --hparams_set=$HPARAMS   --output_dir=$TRAIN_DIR   --t2t_usr_dir=$USR_DIR   --beam_size=$BEAM_SIZE   --model=$MODEL')


# ##4.2. Test the BLEU score
# The BLEU score for all translations: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_bleu.py#L68
# 

# In[ ]:


# get_ipython().system('t2t-bleu    --translations_dir=$TRANSLATIONS_DIR    --model_dir=$TRAIN_DIR    --data_dir=$DATA_DIR    --problem=$PROBLEM    --hparams_set=$HPARAMS    --source=$SOURCE_TEST_TRANSLATE_DIR    --reference=$REFERENCE_TEST_TRANSLATE_DIR    --event_dir=$EVENT_DIR')


# #5. Prediction of sentence
# 

# ##5.1. Predict with terminal
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_decoder.py

# In[ ]:


# get_ipython().system('echo "the business of the house" > "inputs.en"')
# get_ipython().system('echo -e "les affaires de la maison" > "reference.fr" # You can add other references')

# get_ipython().system('t2t-decoder   --data_dir=$DATA_DIR   --problem=$PROBLEM   --model=$MODEL   --hparams_set=$HPARAMS   --output_dir=$TRAIN_DIR   --decode_hparams="beam_size=1,alpha=$ALPHA"   --decode_from_file="inputs.en"   --decode_to_file="outputs.fr"')

# See the translations
# get_ipython().system('cat outputs.fr')


# ##5.2. Predict with code

# In[ ]:
##

import tensorflow as tf

#After training the model, re-run the environment but run this code in first, then predict.

tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys


# In[ ]:


#Config

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
import numpy as np

enfr_problem = problems.problem(PROBLEM)

# Copy the vocab file locally so we can encode inputs and decode model outputs
vocab_name = "vocab.translate_enfr_wmt32k.32768.subwords"
vocab_file = os.path.join(DATA_DIR, vocab_name)

# Get the encoders from the problem
encoders = enfr_problem.feature_encoders(DATA_DIR)

ckpt_path = tf.train.latest_checkpoint(os.path.join(TRAIN_DIR))
print(ckpt_path)

def translate(inputs):
  encoded_inputs = encode(inputs)
  with tfe.restore_variables_on_create(ckpt_path):
    model_output = translate_model.infer(encoded_inputs)["outputs"]
  return decode(model_output)

def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["inputs"].decode(np.squeeze(integers))


# In[ ]:


#Predict 

hparams = trainer_lib.create_hparams(HPARAMS, data_dir=DATA_DIR, problem_name=PROBLEM)
translate_model = registry.model(MODEL)(hparams, Modes.PREDICT)

inputs = "the aniamal didn't cross the river because it was too tired"
ref = "l'animal n'a pas traversé la rue parcequ'il etait trop fatigué" ## this just a reference for evaluate the quality of the traduction
outputs = translate(inputs)

print("Inputs: %s" % inputs)
print("Outputs: %s" % outputs)

file_input = open("outputs.fr","w+")
file_input.write(outputs)
file_input.close()

file_output = open("reference.fr","w+")
file_output.write(ref)
file_output.close()


# ##5.3. Evaluate the BLEU Score
# BLEU score for a sequence translation: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_bleu.py#L24

# In[ ]:


# get_ipython().system('t2t-bleu     --translation=outputs.fr     --reference=reference.fr')


# #6. Attention visualization
# We need to have a predicted sentence with code.

# ##6.1. Attention utils
# 

# In[ ]:


from tensor2tensor.visualization import attention
from tensor2tensor.data_generators import text_encoder

SIZE = 35

def encode_eval(input_str, output_str):
  inputs = tf.reshape(encoders["inputs"].encode(input_str) + [1], [1, -1, 1, 1])  # Make it 3D.
  outputs = tf.reshape(encoders["inputs"].encode(output_str) + [1], [1, -1, 1, 1])  # Make it 3D.
  return {"inputs": inputs, "targets": outputs}

def get_att_mats():
  enc_atts = []
  dec_atts = []
  encdec_atts = []

  for i in range(hparams.num_hidden_layers):
    enc_att = translate_model.attention_weights[
      "transformer/body/encoder/layer_%i/self_attention/multihead_attention/dot_product_attention" % i][0]
    dec_att = translate_model.attention_weights[
      "transformer/body/decoder/layer_%i/self_attention/multihead_attention/dot_product_attention" % i][0]
    encdec_att = translate_model.attention_weights[
      "transformer/body/decoder/layer_%i/encdec_attention/multihead_attention/dot_product_attention" % i][0]
    enc_atts.append(resize(enc_att))
    dec_atts.append(resize(dec_att))
    encdec_atts.append(resize(encdec_att))
  return enc_atts, dec_atts, encdec_atts

def resize(np_mat):
  # Sum across heads
  np_mat = np_mat[:, :SIZE, :SIZE]
  row_sums = np.sum(np_mat, axis=0)
  # Normalize
  layer_mat = np_mat / row_sums[np.newaxis, :]
  lsh = layer_mat.shape
  # Add extra dim for viz code to work.
  layer_mat = np.reshape(layer_mat, (1, lsh[0], lsh[1], lsh[2]))
  return layer_mat

def to_tokens(ids):
  ids = np.squeeze(ids)
  subtokenizer = hparams.problem_hparams.vocabulary['targets']
  tokens = []
  for _id in ids:
    if _id == 0:
      tokens.append('<PAD>')
    elif _id == 1:
      tokens.append('<EOS>')
    elif _id == -1:
      tokens.append('<NULL>')
    else:
        tokens.append(subtokenizer._subtoken_id_to_subtoken_string(_id))
  return tokens

# def call_html():
#   import IPython
#   # display(IPython.core.display.HTML('''
#         <script src="/static/components/requirejs/require.js"></script>
#         <script>
#           requirejs.config({
#             paths: {
#               base: '/static/base',
#               "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
#               jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
#             },
#           });
#         </script>
#         '''))


# ##6.2 Display Attention

# In[ ]:


import numpy as np

# Convert inputs and outputs to subwords

inp_text = to_tokens(encoders["inputs"].encode(inputs))
out_text = to_tokens(encoders["inputs"].encode(outputs))

hparams = trainer_lib.create_hparams(HPARAMS, data_dir=DATA_DIR, problem_name=PROBLEM)

# Run eval to collect attention weights
example = encode_eval(inputs, outputs)
with tfe.restore_variables_on_create(tf.train.latest_checkpoint(ckpt_path)):
  translate_model.set_mode(Modes.EVAL)
  translate_model(example)
# Get normalized attention weights for each layer
enc_atts, dec_atts, encdec_atts = get_att_mats()

# call_html()
attention.show(inp_text, out_text, enc_atts, dec_atts, encdec_atts)


# #7. Export the model
# For more information: https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/serving

# In[ ]:


#export Model
# get_ipython().system('t2t-exporter   --data_dir=$DATA_DIR   --output_dir=$TRAIN_DIR   --problem=$PROBLEM   --model=$MODEL   --hparams_set=$HPARAMS   --decode_hparams="beam_size=1,alpha=$ALPHA"   --export_dir=$EXPORT_DIR')


# #8.Load pretrained model from Google Storage
# We use the pretrained model En-De translation.

# ##8.1. See existing content storaged

# In[ ]:


print("checkpoint: ")
# get_ipython().system('gsutil ls "gs://tensor2tensor-checkpoints"')

print("data: ")
# get_ipython().system('gsutil ls "gs://tensor2tensor-data"')


# ##8.2. Init model

# In[ ]:


PROBLEM_PRETRAINED = "translate_ende_wmt32k"
MODEL_PRETRAINED = "transformer" 
HPARAMS_PRETRAINED = "transformer_base"


# ##8.3. Load content from google storage

# In[ ]:


import tensorflow as tf
import os


DATA_DIR_PRETRAINED = os.path.expanduser("/t2t/data_pretrained")
CHECKPOINT_DIR_PRETRAINED = os.path.expanduser("/t2t/checkpoints_pretrained")

tf.gfile.MakeDirs(DATA_DIR_PRETRAINED)
tf.gfile.MakeDirs(CHECKPOINT_DIR_PRETRAINED)


gs_data_dir = "gs://tensor2tensor-data/"
vocab_name = "vocab.translate_ende_wmt32k.32768.subwords"
vocab_file = os.path.join(gs_data_dir, vocab_name)

gs_ckpt_dir = "gs://tensor2tensor-checkpoints/"
ckpt_name = "transformer_ende_test"
gs_ckpt = os.path.join(gs_ckpt_dir, ckpt_name)

TRAIN_DIR_PRETRAINED = os.path.join(CHECKPOINT_DIR_PRETRAINED, ckpt_name)

# get_ipython().system('gsutil cp {vocab_file} {DATA_DIR_PRETRAINED}')
# get_ipython().system('gsutil -q cp -R {gs_ckpt} {CHECKPOINT_DIR_PRETRAINED}')

CHECKPOINT_NAME_PRETRAINED = tf.train.latest_checkpoint(TRAIN_DIR_PRETRAINED) # for translate with code


# ##8.4. Translate

# In[ ]:


# get_ipython().system('echo "the business of the house" > "inputs.en"')
# get_ipython().system('echo -e "das Geschäft des Hauses" > "reference.de"')

# get_ipython().system('t2t-decoder   --data_dir=$DATA_DIR_PRETRAINED   --problem=$PROBLEM_PRETRAINED    --model=$MODEL_PRETRAINED    --hparams_set=$HPARAMS_PRETRAINED   --output_dir=$TRAIN_DIR_PRETRAINED    --decode_hparams="beam_size=1"   --decode_from_file="inputs.en"   --decode_to_file="outputs.de"')

# See the translations
# get_ipython().system('cat outputs.de')

# get_ipython().system('t2t-bleu     --translation=outputs.de     --reference=reference.de')


# #9.  Add your dataset/problem
# To add a new dataset/problem, subclass Problem and register it with @registry.register_problem. See TranslateEnfrWmt8k for an example: 
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/translate_enfr.py
# 
# ---
# Adding your own components: https://github.com/tensorflow/tensor2tensor#adding-your-own-components
# 
# ---
# 
# See this example: https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/test_data/example_usr_dir

# In[ ]:


from tensor2tensor.utils import registry
#
# @registry.register_problem
# class MyTranslateEnFr(translate_enfr.TranslateEnfrWmt8k):
#
#   def generator(self, data_dir, tmp_dir, train):
#    #your code
#     pass
