from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import umap

# custom imports
from src.visuals import visualize_vector_sequence
import src.get_embeddings as get_embeddings
from src.projectors import project_sequence_umap, project_sequence_pca


# define lame functions idk why they are here
def train_pca(all_embeddings) -> PCA:
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    return pca


def train_umap(all_embeddings) -> umap.UMAP:
    reducer = umap.UMAP(n_components=2)
    reducer.fit(all_embeddings)
    return reducer


# define model and tokenizer
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# load dataset
ds = load_dataset("stanfordnlp/sst2")

train, validation, test = ds["train"], ds["validation"], ds["test"]

# convert to pandas
train_df = pd.DataFrame(train, columns=["sentence", "label"])
validation_df = pd.DataFrame(validation, columns=["sentence", "label"])
test_df = pd.DataFrame(test, columns=["sentence", "label"])

# get embeddings for validation set
validation_sentence_embeddings = get_embeddings.sentence_batch(
    validation_df, tokenizer, model
)

# Visualize embeddings and paths for validation set using PCA  trained on full sentence embeddings

## get PCA transformer for validation set
validation_sentence_pca = PCA(n_components=2).fit(validation_sentence_embeddings)
