import tensorflow as tf

def split_model(model, split_n):
    input1 = tf.keras.layers.Input(batch_shape=(None,None))
    embed_token, embeddings = model.layers[1](input1)
    embed_token_pos = model.layers[2](embed_token)

    next_layers = model.layers[3:split_n*6 + 3]
    tensors = [embed_token_pos]
    for i, layer in enumerate(next_layers):
        if 'Add' in layer.name:
            l = layer([tensors[i-2], tensors[i]])
        else:
            l = layer(tensors[i])
        tensors.append(l)

    m1 = tf.keras.Model(inputs=input1, outputs=[l, embeddings])

    input2 = tf.keras.layers.Input(batch_shape=l.shape)
    input2_embd = tf.keras.layers.Input(batch_shape=embeddings.shape)

    next_layers = model.layers[3 + split_n*6:-1]
    tensors = [input2]
    for i, layer in enumerate(next_layers):
        if 'Add' in layer.name:
            l = layer([tensors[i-2], tensors[i]])
        else:
            l = layer(tensors[i])
        tensors.append(l)
    out = model.layers[-1]([l,input2_embd])

    m2 = tf.keras.Model(inputs=[input2, input2_embd], outputs=out)
    return m1, m2
