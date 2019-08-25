#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import molecule_predictionv3_1 as mp


# In[2]:


import tensorflow as tf


# In[3]:


dat = mp.molecule_data_wrapper()


# In[5]:


atom_vector_size = mp.atom_vector_size
num_atoms = mp.num_atoms
num_timesteps = 10
hidden_unit_size = atom_vector_size
batch_size = 64
num_epochs = 2000

batch_gen = dat.train_gen
test_gen = dat.val_gen

dataset = tf.data.Dataset.from_generator(batch_gen.get_training_sample, (tf.float32,tf.float32, tf.float32),
                   output_shapes= (tf.TensorShape([num_atoms,atom_vector_size]),
                                   tf.TensorShape([num_atoms,num_atoms]),
                                   tf.TensorShape([mp.prediction_vector_size])))                                                     
dataset = dataset.shuffle(buffer_size = batch_size*10) 
dataset = dataset.repeat(num_epochs).batch(batch_size)
dataset = dataset.prefetch(buffer_size = 100)

testset = tf.data.Dataset.from_generator(test_gen.get_training_sample, (tf.float32,tf.float32, tf.float32),
                   output_shapes= (tf.TensorShape([num_atoms,atom_vector_size]),
                                   tf.TensorShape([num_atoms,num_atoms]),
                                   tf.TensorShape([mp.prediction_vector_size])))                                                      
testset = testset.repeat(num_epochs).batch(batch_size)
testset = testset.prefetch(buffer_size = 100)


data_source =  dataset.make_one_shot_iterator()
test_source =  testset.make_one_shot_iterator()

prob = tf.placeholder_with_default(1.0, shape=())
in_keep_prob = tf.placeholder_with_default(1.0, shape=())
out_keep_prob = tf.placeholder_with_default(1.0, shape=())

handle = tf.placeholder(tf.string,name="handle_placeholder", shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, data_source.output_types)

atoms_vector_batch, ad_matrix , batch_y = iterator.get_next()
batchSize = tf.shape(atoms_vector_batch)[0]


# In[6]:


with tf.variable_scope('message_transform_network'):
    hidden1 = tf.keras.layers.Dense(1024,activation='tanh')
    hidden1.build((None,atom_vector_size))
    hidden2 = tf.keras.layers.Dense(1024,activation='tanh')
    hidden2.build((None,1024))
    out_message = tf.keras.layers.Dense(atom_vector_size)
    out_message.build((None, 1024))
def apply_edge_neural_network_transform_to_batch(batch_input):
    input = tf.reshape(batch_input, [-1, atom_vector_size])
    out = apply_edge_neural_network_transform(input)
    return tf.reshape(out, [batchSize, num_atoms, atom_vector_size])    
def apply_edge_neural_network_transform(input):
    global prob
    return out_message.apply(
        hidden2.apply(
            tf.nn.dropout(hidden1.apply(input),keep_prob=prob)))


# In[7]:


def get_messages_batch(ordered_atoms_vector_batch, adjancency_matrix_batch):
    tranformed_batch = apply_edge_neural_network_transform_to_batch(
        ordered_atoms_vector_batch)
    return tf.matmul(adjancency_matrix_batch,tranformed_batch)


# In[8]:


with tf.variable_scope("RecusiveUnitLTSM"):
    shared_lstm_cell =  tf.keras.layers.GRUCell(hidden_unit_size);                                                     
def iterate_time_step(states_vectors_t, messages):
    (outputs_t, states_vectors_t_1) =         shared_lstm_cell(messages, states_vectors_t)
    return outputs_t, states_vectors_t_1


# In[9]:


initial_state1 = tf.Variable(np.zeros(shape=(1,hidden_unit_size)),
                            trainable=True,dtype=tf.float32);
initial_state2 = tf.Variable(np.zeros(shape=(1,hidden_unit_size)),
                            trainable=True,dtype=tf.float32);
initial_states1 = tf.reshape(tf.tile(initial_state1, (1,num_atoms*batchSize)),
                            [num_atoms*batchSize, hidden_unit_size])
initial_states2 = tf.reshape(tf.tile(initial_state2, (1,num_atoms*batchSize)),
                            [num_atoms*batchSize, hidden_unit_size])
initial_states = [initial_states1,initial_states2]


# In[10]:


def graph_neural_network_batch(atoms_vector_batch, ad_matrix):
    global initial_states   
    outputs_x = atoms_vector_batch
    states_vectors_x = initial_states
    for i in range(num_timesteps):
        messages = get_messages_batch(outputs_x, ad_matrix)
        messages = tf.reshape(messages, [-1, atom_vector_size])
        messages = tf.nn.dropout(messages, keep_prob=in_keep_prob)
        outputs_x, states_vectors_x = iterate_time_step(states_vectors_x,messages)    
        outputs_x = tf.nn.dropout(outputs_x, keep_prob=out_keep_prob)
        outputs_x = tf.reshape(outputs_x,[batchSize,num_atoms,atom_vector_size])
    final_outputs = tf.reshape(outputs_x,[batchSize,num_atoms,atom_vector_size])
    out = tf.reduce_sum(final_outputs,axis=1)
    return out


# In[11]:


final_outputs=graph_neural_network_batch(atoms_vector_batch, ad_matrix)
post_gnn_hidden = tf.keras.layers.Dense(512)(final_outputs)
prediction = tf.keras.layers.Dense(mp.prediction_vector_size)(
    tf.nn.dropout(final_outputs,keep_prob=prob))
with tf.name_scope("loss"):
    loss = tf.losses.mean_squared_error(batch_y, prediction)


# In[12]:


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(
    starter_learning_rate, global_step, 1000000, 0.9, staircase=True)
with tf.name_scope("train"):
    global training_op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)


# In[13]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[14]:


with tf.Session() as sess:
    init.run()
    train_iterator_handle = sess.run(data_source.string_handle())
    test_iterator_handle = sess.run(test_source.string_handle())
    for epoch in range(num_epochs):
        for iteration in range(dat.train_gen.samples// batch_size):
            _,loss_value = sess.run([training_op,loss],
                                    feed_dict=
                                    {handle: train_iterator_handle,
                                     prob: 0.5,
                                     in_keep_prob: 0.8,
                                     out_keep_prob: 0.8})  
            if iteration % 200 == 0:
                print("Epoch " + str(epoch) + " Step " + str(iteration) 
                      + "Train Loss " + str(loss_value))
        val_accuracy = 0
        val_step = 1
        for iteration in range(dat.val_gen.samples// batch_size):
            [loss_value] = sess.run([loss],feed_dict={handle: test_iterator_handle})        
            val_accuracy = val_accuracy + loss_value
            val_step = val_step + 1
        val_accuracy = val_accuracy/val_step
        print("Epoch " + str(epoch) + "Validation Loss " + str(val_accuracy))
        if epoch % 50 == 0:
            save_path = saver.save(sess, "../models/graph_model" + str(epoch) + ".ckpt")            


# In[ ]:




