import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dimension, output_dimension,
        context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (output_dimension % num_heads == 0), \
            "output_dimension must be divisible by num_heads"
            
        self.output_dimension = output_dimension
        self.num_heads = num_heads
        
        self.head_dimension = output_dimension // num_heads
        
        self.W_query = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_key = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        
        self.out_proj = nn.Linear(output_dimension, output_dimension)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        batch, num_tokens, input_dimension = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dimension)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dimension)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dimension)


        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            batch, num_tokens, self.output_dimension
        )
        context_vec = self.out_proj(context_vec)
        return context_vec