import keras
import matplotlib.pyplot as plt
from keras import layers, ops
import numpy as np
import tensorflow as tf

class CrossAttention(keras.layers.Layer):
  def __init__(self,num_heads, embed_dim, **kwargs):
    super().__init__()
    self.mha = keras.layers.MultiHeadAttention(num_heads, embed_dim)
    self.add = keras.layers.Add() 
    self.layernorm = keras.layers.LayerNormalization()

  def call(self, x, y):
    attn, attention_scores = self.mha(
             query=x, value=y,
             return_attention_scores=True)

    self.last_attention_scores = attention_scores

    x = self.add([x, attn])
    return self.layernorm(x)

class BitLinear(keras.Layer):
    def __init__(self, units=1, input_shape=None):
        super(BitLinear, self).__init__()
        self.units = units
        self.input_dim = input_shape
        if not input_shape is None:
            # self.w = self.add_weight(shape=( self.units, input_dim[-1],), initializer=self.initialize_weights, trainable=True, dtype="int8")
            self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.initialize_weights, trainable=True, dtype="float16")


            self.b = self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True, dtype="float16"
            )    
    def build(self, input_shape):
        if self.input_dim is None:
            self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.initialize_weights, trainable=True, dtype="float16")
            self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, dtype="float16")    

    def q_fun(self, x, a = -1., b = 1., dtype = "float16"):
        ls = x
        return ops.cast(ops.maximum(a, ops.minimum(tf.round(x), b)), dtype=dtype)

    def initialize_weights(self, shape, dtype=None):
        weights = np.random.randn(shape[0], shape[1])
        return self.q_fun(weights, dtype=dtype)

    def call(self, inputs):
        w = ops.cast(self.w, "float32")
        b = ops.cast(self.b, "float32")
        return ops.matmul(inputs, w) + b

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class FeedForward(layers.Layer):
    def __init__(self, units, dropout_rate=0.4):
        super().__init__()
        self.seq = keras.Sequential([
            keras.layers.Dense(units=units*2),
            keras.layers.ReLU(),
            keras.layers.Dense(units=units),
            keras.layers.Dropout(rate=dropout_rate),
        ])

        self.layernorm = keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.seq(x)
        return self.layernorm(x)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # The ffn can be either a standard feedforward network or a switch
        # layer with a Mixture of Experts.
        self.ffn = ffn
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class TransformerBlock_Cross(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.4):
        super().__init__()
        self.att = CrossAttention(num_heads, embed_dim)
        self.ffn = FeedForward(ff_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):

        in_seq, txt_seq = inputs
        
        input_shape = ops.shape(txt_seq)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        # causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attention_output = self.att(txt_seq, in_seq)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

def load_balanced_loss(router_probs, expert_mask):
    # router_probs [tokens_per_batch, num_experts] is the probability assigned for
    # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
    # the expert with the highest router probability in one−hot format.

    num_experts = ops.shape(expert_mask)[-1]
    # Get the fraction of tokens routed to each expert.
    # density is a vector of length num experts that sums to 1.
    density = ops.mean(expert_mask, axis=0)
    # Get fraction of probability mass assigned to each expert from the router
    # across all tokens. density_proxy is a vector of length num experts that sums to 1.
    density_proxy = ops.mean(router_probs, axis=0)
    # Want both vectors to have uniform allocation (1/num experts) across all
    # num_expert elements. The two vectors will be pushed towards uniform allocation
    # when the dot product is minimized.
    loss = ops.mean(density_proxy * density) * ops.cast((num_experts**2), "float32")
    return loss

def create_feedforward_network(ff_dim, embed_dim, name=None):
    return keras.Sequential(
        [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim)], name=name
    )

class Router(layers.Layer):
    def __init__(self, num_experts, expert_capacity):
        self.num_experts = num_experts
        self.route = layers.Dense(units=num_experts)
        self.expert_capacity = expert_capacity
        super().__init__()

    def call(self, inputs, training=False):
        # inputs shape: [tokens_per_batch, embed_dim]
        # router_logits shape: [tokens_per_batch, num_experts]
        router_logits = self.route(inputs)

        if training:
            # Add noise for exploration across experts.
            router_logits += keras.random.uniform(
                shape=router_logits.shape, minval=0.9, maxval=1.1
            )
        # Probabilities for each token of what expert it should be sent to.
        router_probs = keras.activations.softmax(router_logits, axis=-1)
        # Get the top−1 expert for each token. expert_gate is the top−1 probability
        # from the router for each token. expert_index is what expert each token
        # is going to be routed to.
        expert_gate, expert_index = ops.top_k(router_probs, k=1)
        # expert_mask shape: [tokens_per_batch, num_experts]
        expert_mask = ops.one_hot(expert_index, self.num_experts)
        # Compute load balancing loss.
        aux_loss = load_balanced_loss(router_probs, expert_mask)
        self.add_loss(aux_loss)
        # Experts have a fixed capacity, ensure we do not exceed it. Construct
        # the batch indices, to each expert, with position in expert make sure that
        # not more that expert capacity examples can be routed to each expert.
        position_in_expert = ops.cast(
            ops.cumsum(expert_mask, axis=0) * expert_mask, "int32"
        )
        # Keep only tokens that fit within expert capacity.
        expert_mask *= ops.cast(
            ops.less(ops.cast(position_in_expert, "int32"), self.expert_capacity),
            "float32",
        )
        expert_mask_flat = ops.sum(expert_mask, axis=-1)
        # Mask out the experts that have overflowed the expert capacity.
        expert_gate *= expert_mask_flat
        # Combine expert outputs and scaling with router probability.
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        combined_tensor = ops.expand_dims(
            expert_gate
            * expert_mask_flat
            * ops.squeeze(ops.one_hot(expert_index, self.num_experts), 1),
            -1,
        ) * ops.squeeze(ops.one_hot(position_in_expert, self.expert_capacity), 1)
        # Create binary dispatch_tensor [tokens_per_batch, num_experts, expert_capacity]
        # that is 1 if the token gets routed to the corresponding expert.
        dispatch_tensor = ops.cast(combined_tensor, "float32")

        return dispatch_tensor, combined_tensor
    
class Switch(layers.Layer):
    def __init__(
        self, num_experts, embed_dim, ff_dim, num_tokens_per_batch, capacity_factor=1, maxlen = 128
    ):
        self.maxlen = maxlen
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.experts = [
            create_feedforward_network(ff_dim, embed_dim) for _ in range(num_experts)
        ]

        self.expert_capacity = num_tokens_per_batch // self.num_experts
        self.router = Router(self.num_experts, self.expert_capacity)
        super().__init__()

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        num_tokens_per_example = ops.shape(inputs)[1]

        # inputs shape: [num_tokens_per_batch, embed_dim]
        inputs = ops.reshape(inputs, [self.maxlen, self.embed_dim])
        # dispatch_tensor shape: [expert_capacity, num_experts, tokens_per_batch]
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        dispatch_tensor, combine_tensor = self.router(inputs)
        # expert_inputs shape: [num_experts, expert_capacity, embed_dim]
        expert_inputs = ops.einsum("ab,acd->cdb", inputs, dispatch_tensor)
        expert_inputs = ops.reshape(
            expert_inputs, [self.num_experts, self.expert_capacity, self.embed_dim]
        )
        # Dispatch to experts
        expert_input_list = ops.unstack(expert_inputs, axis=0)
        expert_output_list = [
            self.experts[idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
        # expert_outputs shape: [expert_capacity, num_experts, embed_dim]
        expert_outputs = ops.stack(expert_output_list, axis=1)
        # expert_outputs_combined shape: [tokens_per_batch, embed_dim]
        expert_outputs_combined = ops.einsum(
            "abc,xba->xc", expert_outputs, combine_tensor
        )
        # output shape: [batch_size, num_tokens_per_example, embed_dim]
        outputs = ops.reshape(
            expert_outputs_combined,
            [batch_size, num_tokens_per_example, self.embed_dim],
        )
        return outputs