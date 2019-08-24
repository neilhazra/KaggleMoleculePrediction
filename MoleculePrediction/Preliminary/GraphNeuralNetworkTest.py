#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import math
import molecule_predictionv2_0 as mp


# In[5]:


import tensorflow as tf


# In[6]:


dat = mp.molecule_prediction_data_wrapper()


# In[21]:


atom_vector_size = 1024
num_timesteps = 5
hidden_unit_size = 1024
batchSize = 32
num_epochs = 200
batch_gen = dat.batch_gen
dataset = tf.data.Dataset.    from_generator(batch_gen.generate, (tf.float32,tf.float32),
                   output_shapes= (tf.TensorShape([30,1054]), 
                                   tf.TensorShape([3])))                                                     
dataset = dataset.shuffle(buffer_size = batchSize*10) 
dataset = dataset.repeat(num_epochs).batch(batchSize)
dataset = dataset.prefetch(buffer_size = 2)
data_source =  dataset.make_one_shot_iterator()
batch_in, batch_y = data_source.get_next()


# In[8]:


with tf.variable_scope('message_transform_network'):
    hidden1 = tf.keras.layers.Dense(2048,activation='relu')
    hidden1.build((None,atom_vector_size))
    hidden2 = tf.keras.layers.Dense(2048,activation='tanh')
    hidden2.build((None,2048))
    hidden3 = tf.keras.layers.Dense(1024,activation='relu')
    hidden3.build((None,2048))
    out_message = tf.keras.layers.Dense(atom_vector_size)
    out_message.build((None, 1024))
def apply_edge_neural_network_transform(input):
    return out_message.apply(hidden3.apply(hidden2.apply(hidden1.apply(input))))


# In[9]:


def get_messages(ordered_atoms_vector, adjacency_matrix):
    transformed = apply_edge_neural_network_transform(ordered_atoms_vector)
    return tf.matmul(adjacency_matrix,transformed)


# In[10]:


with tf.variable_scope("RecusiveUnitLTSM"):
    shared_lstm_cell =  tf.keras.layers.LSTMCell(hidden_unit_size);
def iterate_time_step(states_vectors_t, messages):
    (outputs_t, states_vectors_t_1) =         shared_lstm_cell(messages, states_vectors_t)
    return outputs_t, states_vectors_t_1


# In[11]:


initial_state1 = tf.Variable(np.random.normal(size=(1,hidden_unit_size)),
                            trainable=True,dtype=tf.float32);
initial_state2 = tf.Variable(np.random.normal(size=(1,hidden_unit_size)),
                            trainable=True,dtype=tf.float32);
initial_states1 = tf.reshape(tf.tile(initial_state1, (1,tf.shape(batch_in)[1])),
                            [tf.shape(batch_in)[1], hidden_unit_size])
initial_states2 = tf.reshape(tf.tile(initial_state2, (1,tf.shape(batch_in)[1])),
                            [tf.shape(batch_in)[1], hidden_unit_size])
initial_states = (initial_states1,initial_states2)


# In[12]:


def extract_atom_vectors_ad_matrix(input_mat):
    num_atoms = tf.shape(input_mat)[0]
    return tf.slice(input_mat,[0,0],[num_atoms,atom_vector_size]),           tf.slice(input_mat,[0,atom_vector_size],[num_atoms,num_atoms])


# In[13]:


def graph_neural_network(concatenated_input_mat):
    global initial_states
    initial_output,ad_matrix =         extract_atom_vectors_ad_matrix(concatenated_input_mat)
    outputs_x = initial_output
    states_vectors_x = initial_states
    for i in range(num_timesteps):
        messages = get_messages(outputs_x, ad_matrix)
        outputs_x, states_vectors_x = iterate_time_step(states_vectors_x,messages)    
    final_outputs = outputs_x
    out = tf.reduce_sum(final_outputs, axis=0)
    return out 


# In[25]:


final_outputs=tf.map_fn(graph_neural_network,batch_in)
prediction = tf.keras.layers.Dense(3)(final_outputs)
with tf.name_scope("loss"):
    loss = tf.losses.mean_squared_error(batch_y, prediction)


# In[15]:


learning_rate = 0.001
with tf.name_scope("train"):
    global training_op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)


# In[27]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(num_epochs):
        for iteration in range(batch_gen.samples// batchSize):
            _,loss_value = sess.run([training_op,loss])
            if iteration % 500 == 0:
                print("Epoch " + str(epoch) + " Step " + str(iteration) + " loss " + str(loss_value))
        if epoch % 20 == 0:
            save_path = saver.save(sess, "../models/graph_model_" + str(epoch) + ".ckpt")


# In[ ]:




