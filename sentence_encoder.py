import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# while running on your PC , comment the below line
embed = hub.Module("../sentence_wise_email/module/module_useT")

# while running on your PC, un-comment the below 2 lines
#module_url = "https://tfhub.dev/google/universal-sentence-encoder/1?tf-hub-format=compressed"
#embed = hub.Module(module_url)

# We are running this for the training data set
data = pd.read_csv('cleanedtrain2.csv', delimiter=',')
data1 = data[['id','title1_en','title2_en','label']]
data1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
data1 = data1.head(1000)
data2 = data1

tf.logging.set_verbosity(tf.logging.ERROR)

similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.compat.v1.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.compat.v1.tables_initializer())

    message_embeddings1 = session.run(embed(data2['title1_en']))
    message_embeddings2 = session.run(embed(data1['title2_en']))
    
    np.save('test3.npy', message_embeddings1)
    np.save('test4.npy', message_embeddings2)

# Similarly run the above code for the validation set and testing data set