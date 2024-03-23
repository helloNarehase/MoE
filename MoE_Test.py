import math
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import layers, ops
from module import *
import param

params = param.param()

def create_classifier():
    inputs = layers.Input(shape=(None,))
    embedding_layer = GPT_Embedding(
        params.maxlen, params.vocaps, params.embed_dim
    )
    x = embedding_layer(inputs)
    for i in range(params.layers): 
        switch = Switch(params.num_experts, params.embed_dim, params.ffn_dim, params.maxlen)
        transformer_block = TransformerBlock(params.embed_dim // params.num_heads, params.num_heads, switch)
        x = transformer_block(x)
    outputs = layers.Dense(params.vocaps)(x)
    classifier = keras.Model(inputs=inputs, outputs=outputs)
    return classifier

model = create_classifier()
model.summary()