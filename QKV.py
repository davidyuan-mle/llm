"""
implement the QKV self-attention mechanism

input:  a batch of text sequences
output: self-attention scores

"""

import tiktoken
import numpy as np

# encoding the text sequences
# r50k_base is ChatGPT 3's encoding scheme
def tokenize(text, encoding_name='r50k_base'):
    tokenizer = tiktoken.get_encoding(encoding_name)
    token_ids = tokenizer.encode(text)
    return np.array(token_ids), tokenizer

# encoding the sequence positions 
def position_embedding(seq_len, n_embedding):
    pos_enc = np.random.randn(seq_len, n_embedding) / np.sqrt(n_embedding)
    return pos_enc 

# calculate QKV 
def get_self_attention_scores(token_ids, vocab_size, n_embedding=64, d_k=64, d_v=64):
    """
    Compute self-attention scores for the input token IDs.
    :param token_ids: Array of token IDs (seq_len,)
    :param vocab_size: Size of the vocabulary
    :param n_embedding: Embedding dimension
    :param d_k: Query/key dimension
    :return: Attention scores (1, seq_len, seq_len)
    """
    seq_len = len(token_ids)
    
    # Initialize token embeddings
    embedding_matrix = np.random.randn(vocab_size, n_embedding) / np.sqrt(n_embedding)
    token_embeddings = embedding_matrix[token_ids]  # (seq_len, n_embedding)
    
    # Add positional encodings
    pos_enc = position_embedding(seq_len, n_embedding)  # (seq_len, n_embedding)
    embeddings = token_embeddings + pos_enc  # (seq_len, n_embedding)

    # Initialize Q, K weight matrices
    W_q = np.random.randn(n_embedding, d_k) / np.sqrt(n_embedding)
    W_k = np.random.randn(n_embedding, d_k) / np.sqrt(n_embedding)
    W_v = np.random.randn(n_embedding, d_v) / np.sqrt(n_embedding)
    
    # Compute Q, K
    Q = embeddings @ W_q  # (seq_len, d_k)
    K = embeddings @ W_k  # (seq_len, d_k)
    V = embeddings @ W_v  # (seq_len, d_v)
    
    # Compute attention scores: (Q * K^T) / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)
    
    # Apply softmax to get attention probabilities
    exp_scores = np.exp(scores)
    attention_scores = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)  # (seq_len, seq_len)

    # update V 
    V = attention_scores @ V
    
    return attention_scores, V


text = 'I have two pets: one is a dog, the other is a cat.'
token_ids, tokenizer = tokenize(text)
attention_scores, V = get_self_attention_scores(token_ids, vocab_size=tokenizer.n_vocab, n_embedding=64, d_k=64, d_v=64)

print(f'text size: {len(text)}')
print(f'token size: {len(token_ids)}')
print(f'self attention score shape: {attention_scores.shape}')
print(f'matrix V shape: {V.shape}')
