#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.text_classifier import DataLoader
from tflite_model_maker.config import QuantizationConfig

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')


# In[2]:


# Remove and stopwords and special characters
from bs4 import BeautifulSoup
import string

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

table = str.maketrans('', '', string.punctuation)


# In[3]:


dataset = pd.read_csv('/tmp/spamDetection/spam.csv', encoding="ISO-8859-1")


# In[4]:


dataset.head()


# In[5]:


sentences = [] 
labels = []
for index, item in dataset.iterrows():
    sentence = item['v2'].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append("positive") if item['v1'] == "spam" else labels.append("negative")


# In[6]:


# Create new pandas dataframe and save as csv
dict = {'comments': sentences, 'label': labels}
df = pd.DataFrame(dict)
print(df)


# In[7]:


df.to_csv('/tmp/spamDetection/cleanuped_spam.csv', index=False)


# In[8]:


# Load data from csv and split to test, train
#spec = model_spec.get('average_word_vec')
spec = model_spec.get('mobilebert_classifier')
#spec.num_words = 2000
#spec.seq_len = 20
#spec.wordvec_dim = 7
spec.dropout_rate = 0.2
spec.learning_rate = 0.0001


# In[9]:


data = DataLoader.from_csv(
    filename="/tmp/spamDetection/cleanuped_spam.csv",
    text_column='comments',
    label_column='label',
    model_spec=spec,
    delimiter=',',
    shuffle=True,
    is_training=True)

train_data, test_data = data.split(0.9)


# In[10]:


# Build the model
# model = text_classifier.create(train_data, model_spec=spec, epochs=50, validation_data=test_data)
model = text_classifier.create(train_data, model_spec=spec, epochs=10)


# In[11]:


loss, acc = model.evaluate(test_data)


# In[12]:


# Save model
export_dir = '/tmp/saved_model/spam/'
#model.export(export_dir=export_dir, export_format=ExportFormat.SAVED_MODEL)


# In[13]:


# Reduce model size by quantization
# https://www.tensorflow.org/lite/performance/post_training_quantization
# https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/config/QuantizationConfig#for_dynamic
#config = QuantizationConfig.for_float16()


# In[14]:


# This will export to TFLite format with the model only. 
# if you see a .json file in this directory, it is NOT the JSON model for TFJS
# See below for how to generate that.
# Please note that if you run this cell to create the tflite model then the 
# export to TFJS will fail. You'll need to rerun the model training first
#model.export(export_dir=export_dir, tflite_filename='model_fp16.tflite', quantization_config=config)
model.export(export_dir=export_dir)


# In[15]:


# If you want the labels and the vocab, for example for iOS, you can use this
model.export(export_dir=export_dir, export_format=[ExportFormat.LABEL, ExportFormat.VOCAB])

# You can find your files in colab by clicking the 'folder' tab to the left of
# this code window, and then navigating 'up' a directory to find the root
# directory listing -- and from there you should see /mm_spam/


# In[ ]:


#Evaluating Exported Model
loss, acc = model.evaluate_tflite(export_dir + 'model.tflite', test_data)


# In[ ]:


# Use this section for export to TFJS
# Please note that if you run the above cell to create the tflite model then the 
# export to TFJS will fail. You'll need to rerun the model training first
#model.export(export_dir="/tmp/saved_model/spam/tfjs/", export_format=[ExportFormat.TFJS, ExportFormat.LABEL, ExportFormat.VOCAB])

