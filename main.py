import os
import torch
import tiktoken

from data_preparation.dataset import GPTDataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from data_preparation.data_embedding import DataEmbedding

load_dotenv()

batch_size = int(os.getenv("DATALOADER_BATCH_SIZE"))
max_length = int(os.getenv("DATALOADER_MAX_LENGTH"))
stride = int(os.getenv("DATALOADER_STRIDE"))
shuffle = bool(os.getenv("DATALOADER_SHUFFLE") == True)

output_dimensions = int(os.getenv("EMBEDDING_OUTPUT_DIM", 768))


tokenizer = tiktoken.get_encoding("gpt2")# Default to 768 if not set


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
    
dataloader = create_dataloader(raw_text, batch_size, max_length, stride, shuffle)
data_embedding = DataEmbedding(max_length, output_dimensions, tokenizer.n_vocab)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

input_embeddings = data_embedding.embed_data(inputs)
print("Output shape:", input_embeddings.shape)