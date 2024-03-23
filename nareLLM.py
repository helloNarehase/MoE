import keras
import matplotlib.pyplot as plt
from keras import layers, ops
import numpy as np
from module import *

vocab_size = 20000  # Only consider the top 20k words
maxlen = 128  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer
transformer_layers = 12
def getLLM_Layer():

    inputs = layers.Input(shape=(None,), dtype="int32")
    embedding_layer = GPT_Embedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    for i in range(transformer_layers):
        transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim, fp16= False)
        x = transformer_block(x)
    
    outputs = layers.Dense(vocab_size)(x)
    return inputs, outputs

def createLLM():
    inputs = layers.Input(shape=(None,), dtype="int32")
    embedding_layer = GPT_Embedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    for i in range(transformer_layers):
        transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim, fp16= False)
        x = transformer_block(x)
    
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model

# model = createLLM()
# model.summary()

