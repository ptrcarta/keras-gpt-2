import os
from timeit import default_timer as timer
os.environ['TF_KERAS'] = '1'

#1.5b
#hparams = {
#    "n_vocab": 50257,
#    "n_ctx": 1024,
#    "n_embd": 1600,
#    "n_head": 25,
#    "n_layer": 48
#}

#774M
hparams = {                                                                                  
    "n_vocab": 50257,                                                                
    "n_ctx": 1024,                                                                   
    "n_embd": 1280,                                                                  
    "n_head": 20,                                                                    
    "n_layer": 36                                                                    
} 

import keras_gpt_2
#import split
import numpy as np

# inititalize strategies
import tensorflow as tf
from tensorflow.python.tpu import device_assignment  as device_assignment_lib
from tensorflow.python.distribute import tpu_strategy as tpu_lib
if 'COLAB_TPU_ADDR' in os.environ:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
else:
    from resolver import res as resolver

tf.config.experimental_connect_to_cluster(resolver)
topology = tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)

device_assignment = device_assignment_lib.DeviceAssignment.build(
    topology, num_replicas=4)

first_strategy = tpu_lib.TPUStrategy(
    resolver, device_assignment=device_assignment)

# Computation on the 2nd half.
device_assignment2 = device_assignment_lib.DeviceAssignment(
    topology, [[[1, 0, 0]],[[1, 0, 1]],[[1, 1, 0]],[[1, 1, 1]]])

second_strategy = tpu_lib.TPUStrategy(
    resolver, device_assignment=device_assignment2)


print('initialized strategies')
#breakpoint()

split_n=20
with first_strategy.scope():
    m1, l_shape, embd_shape = keras_gpt_2.model.get_model_first(split_n=split_n, **hparams)
    #m1, _ = split.split_model(model, split_n=23)
    #del model
    opt1 = tf.keras.optimizers.Adagrad(learning_rate=1e-4)
with second_strategy.scope():
    m2 = keras_gpt_2.model.get_model_second(split_n=split_n,
            l_shape=l_shape, embeddings_shape=embd_shape,
            **hparams)
    #_, m2 = split.split_model(model, split_n=23)
    #del model
    opt2 = tf.keras.optimizers.Adagrad(learning_rate=1e-4)

print('created models')
#breakpoint()


@tf.function
def first_half(x, output_grads, embd_grads):
    with tf.GradientTape() as tape:
        intermed, embeddings = m1(x)
        intermed_loss = tf.reduce_sum(intermed*output_grads)
        embd_loss = tf.reduce_sum(embeddings*embd_grads)
        grads = tape.gradient(intermed_loss + embd_loss, m1.trainable_variables)
    opt1.apply_gradients(zip(grads, m1.trainable_variables))
    return intermed, embeddings

@tf.function
def second_half(inter, embd):
    with tf.GradientTape() as tape:
        tape.watch([inter, embd])
        output_data = m2([inter, embd])
        loss = tf.reduce_mean(output_data) #fake loss for dev purposes
    local_grads, input_grads, embd_grads= tape.gradient(loss,[m2.trainable_variables, inter, embd])
    opt2.apply_gradients(zip(local_grads, m2.trainable_variables))
    #input_grads, embd_grads = tape.gradient(loss, [inter, embd])

    return output_data, input_grads, embd_grads

# fake input data for now
input_data = np.random.randint(0, 50000, size=(1,1024))

@tf.function
def fake_get_data(): #wrap fake input data into perreplica objects
    intermed, embeds = m1(input_data)
    return input_data, intermed, embeds

@tf.function
def first_iter_intermed_embd():
    return tf.zeros()

#@tf.function
def test():
    per_replica_data, fake_intermed_grad, fake_embeds_grad = first_strategy.experimental_run_v2(fake_get_data)
    intermed1, embd1 = first_strategy.experimental_run_v2(
                    first_half, (per_replica_data, fake_intermed_grad, fake_embeds_grad))
    #breakpoint()
    print('intermed', intermed1.values[0].shape)
    print('embd', embd1.values[0].shape)
    out, intermed_grads1, embd_grads1 = second_strategy.experimental_run_v2(second_half, (intermed1, embd1))
    print('output', out.values[0].shape)
    print('interemd_grads', intermed_grads1.values[0].shape)
    print('embd_grads', embd_grads1.values[0].shape)

    start = timer()
    for i in range(800):
        intermed2, embd2 = first_strategy.experimental_run_v2(
                        first_half, (per_replica_data, intermed_grads1, embd_grads1))
        out, intermed_grads2, embd_grads2 = second_strategy.experimental_run_v2(second_half, (intermed1, embd1))
        intermed1, embd1 = first_strategy.experimental_run_v2(
                        first_half, (per_replica_data, intermed_grads2, embd_grads2))
        #out, intermed_grads1, embd_grads1 = second_strategy.experimental_run_v2(second_half, (intermed2, embd2))
    print('output', out.values[0].shape)
    end = timer()
    print('time:', end - start)
    sum = tf.function(tf.reduce_sum)
    return second_strategy.experimental_run_v2(sum, (out,))

#%%time
out = test()
print(out)

