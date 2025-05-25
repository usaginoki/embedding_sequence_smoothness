from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import os
from src.visuals import visualize_vector_sequence
from src.projectors import project_sequence_umap, project_sequence_pca

# ds = load_dataset("stanfordnlp/sst2")

# print(ds)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

sentence = "I was first attracted by Camus, 'prince of the absurd' when I was 16. Camus still fascinates me, now well beyond what would have been his 100th birthday, and more than 60 years after his premature death in a car crash in Burgundy (it's said that he was found with an unused train ticket in his pocket - he'd planned to go by rail to Paris to rejoin his wife and children, but had accepted at the last minute the offer of a lift from his publisher)."


# Load model directly
def test_pca(sentence, tokenizer, model):
    results_folder = "results/figs"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    vectors_pca, tokens = project_sequence_pca(sentence, tokenizer, model)

    # Visualize the vectors
    fig, ax = visualize_vector_sequence(
        vectors_pca,
        title="Vector Sequence Visualization",
        point_labels=tokens,
        show_points=False,
        point_alpha=0.5,
        arrow_alpha=0.5,
        label_alpha=0.5,
    )

    plt.savefig(
        os.path.join(results_folder, "pca.png"),
        bbox_inches="tight",
        dpi=300,
    )


def test_umap(sentence, tokenizer, model):
    results_folder = "results/figs"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    vectors_umap, tokens = project_sequence_umap(
        sentence,
        tokenizer,
        model,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
    )
    fig, ax = visualize_vector_sequence(
        vectors_umap,
        title="Vector Sequence Visualization",
        point_labels=tokens,
        show_points=False,
        point_alpha=0.5,
        arrow_alpha=0.5,
        label_alpha=0.5,
    )

    plt.savefig(
        os.path.join(results_folder, "umap.png"),
        bbox_inches="tight",
        dpi=300,
    )
