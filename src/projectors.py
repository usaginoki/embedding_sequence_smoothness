import umap
import transformers
import numpy
from sklearn.decomposition import PCA
from .get_embeddings import substrings_single


def project_sequence_umap(
    sentence: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    n_components: int = 2,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
) -> numpy.ndarray:
    embeddings, tokens = substrings_single(sentence, tokenizer, model)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    embeddings_reduced = reducer.fit_transform(embeddings)

    return embeddings_reduced, tokens


def project_sequence_pca(
    sentence: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    n_components: int = 2,
) -> numpy.ndarray:
    embeddings, tokens = substrings_single(sentence, tokenizer, model)
    embeddings_reduced = PCA(n_components=n_components).fit_transform(embeddings)
    return embeddings_reduced, tokens
