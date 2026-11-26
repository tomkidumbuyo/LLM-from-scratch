import torch
import tiktoken

class DataEmbedding:
    def __init__(self, max_length, embedding_output_dim, vocab_size):
        self.context_length = max_length
        self.embedding_output_dim = embedding_output_dim
        self.vocab_size = vocab_size


    def embed_data(self, token_embedding_layer, position_embedding_layer, inputs):
        token_embeddings = token_embedding_layer(inputs)
        position_embeddings = position_embedding_layer(torch.arange(self.context_length))
        input_embeddings = token_embeddings + position_embeddings
        return input_embeddings
    
    
    def create_embedding_layers(self):
        token_embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_output_dim)
        position_embedding_layer = torch.nn.Embedding(self.context_length, self.embedding_output_dim)
        return token_embedding_layer, position_embedding_layer
    
