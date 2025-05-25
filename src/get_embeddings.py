import torch
import numpy
import transformers


def substrings_single(
    sentence: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    return_tokens: bool = True,
) -> numpy.ndarray:
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    print(f"Tokens: {tokens}")
    all_token_substrings = [tokens[i:] for i in range(len(tokens))]
    all_substrings = [
        tokenizer.convert_tokens_to_string(tokens) for tokens in all_token_substrings
    ]
    # Generate embeddings for progressive substrings

    # Generate embedding
    inputs = tokenizer(
        all_substrings, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

        # Get the embedding (using cls token)
        embeddings = outputs[0][:, 0]
    print(f"Embedding shape: {embeddings.shape}")

    if return_tokens:
        return embeddings.numpy(), tokens
    else:
        return embeddings.numpy()


def substrings_batch(df, tokenizer, model):
    all_substrings = []
    for sentence in df["sentence"]:
        tokens = tokenizer.tokenize(sentence)
        for i in range(1, len(tokens)):
            all_substrings.append(tokenizer.convert_tokens_to_string(tokens[:i]))

    inputs = tokenizer(
        all_substrings, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs[0][:, 0]
    return embeddings.numpy()


def sentence_single(sentence, tokenizer, model):
    return substrings_single(sentence, tokenizer, model, return_tokens=False)


def sentence_batch(df, tokenizer, model):
    inputs = tokenizer(
        df["sentence"], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs[0][:, 0]
    return embeddings.numpy()
