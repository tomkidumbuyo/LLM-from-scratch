import os
import torch
import tiktoken

from dataset import GPTDataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv()

batch_size = int(os.getenv("DATALOADER_BATCH_SIZE"))
max_length = int(os.getenv("DATALOADER_MAX_LENGTH"))
stride = int(os.getenv("DATALOADER_STRIDE"))
shuffle = bool(os.getenv("DATALOADER_SHUFFLE") == True)

output_dimensions = int(os.getenv("EMBEDDING_OUTPUT_DIM", 768))  # Default to 768 if not set

tokenizer = tiktoken.get_encoding("gpt2")

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
    

with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
    
dataloader = create_dataloader(raw_text, batch_size=batch_size, max_length=max_length, stride=stride, shuffle=shuffle)


torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(tokenizer.n_vocab, output_dimensions)
print(token_embedding_layer(torch.tensor([3])))


data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


context_length = max_length
position_embedding_layer = torch.nn.Embedding(context_length, output_dimensions)
position_embeddings = position_embedding_layer(torch.arange(context_length))
print(position_embeddings.shape)

input_embeddings = token_embeddings + position_embeddings
print(input_embeddings.shape)