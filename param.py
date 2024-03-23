from dataclasses import dataclass

@dataclass
class param:
    maxlen = 256
    num_experts = 8
    ffn_dim = 768
    layers = 8
    num_heads = 12
    vocaps = 200000
    dropout_rate = 0.2
    embed_dim = 256