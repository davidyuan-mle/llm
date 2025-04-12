"""
implement the QKV self-attention mechanism

input:  a batch of text sequences
output: self-attention scores

"""

import tiktoken
import numpy as np

# encoding the text sequences
# r50k_base is ChatGPT 3's encoding scheme, can use other encoding schemes
# such as gpt2, p50k_base, etc.
def tokenize(text, encoding_name='r50k_base'):
    tokenizer = tiktoken.get_encoding(encoding_name)
    token_ids = tokenizer.encode(text)
    return np.array(token_ids), tokenizer

# encoding the sequence positions 
# can use sinusoidal encoding or learned positional encoding
# here we use learned positional encoding
# the learned positional encoding is a random matrix
# with the same shape as the token embeddings
# the learned positional encoding is initialized with a normal distribution
# with mean 0 and std 1/sqrt(n_embedding)
# the learned positional encoding is updated during training
def position_embedding(seq_len, n_embedding):
    pos_enc = np.random.randn(seq_len, n_embedding) / np.sqrt(n_embedding)
    return pos_enc 

# calculate QKV 
# QKV is the query, key, value matrix
# QKV is used to calculate the self-attention scores
def get_self_attention_scores(token_ids, vocab_size, n_embedding=64, d_k=64, d_v=64):
    """
    Compute self-attention scores for the input token IDs.
    :param token_ids: Array of token IDs (seq_len,)
    :param vocab_size: Size of the vocabulary, for example, 50257
    :param n_embedding: Embedding dimension
    :param d_k: Query/key dimension
    :param d_v: Value dimension
    :return: Attention scores (seq_len, seq_len)
    """
    seq_len = len(token_ids)
    
    # Initialize token embeddings
    embedding_matrix = np.random.randn(vocab_size, n_embedding) / np.sqrt(n_embedding)
    token_embeddings = embedding_matrix[token_ids]  # (seq_len, n_embedding)
    
    # Add positional encodings
    pos_enc = position_embedding(seq_len, n_embedding)  # (seq_len, n_embedding)
    embeddings = token_embeddings + pos_enc  # (seq_len, n_embedding)

    # Initialize Q, K, V weight matrices
    W_q = np.random.randn(n_embedding, d_k) / np.sqrt(n_embedding) 
    W_k = np.random.randn(n_embedding, d_k) / np.sqrt(n_embedding)
    W_v = np.random.randn(n_embedding, d_v) / np.sqrt(n_embedding)
    
    # Compute Q, K, V
    Q = embeddings @ W_q  # (seq_len, d_k)
    K = embeddings @ W_k  # (seq_len, d_k)
    V = embeddings @ W_v  # (seq_len, d_v)
    
    # Compute attention scores: (Q * K^T) / sqrt(d_k)
    # score is the dot product of Q and K
    # score means the similarity between the query and key
    # the higher the score, the more similar the query and key
    # the lower the score, the less similar the query and key
    # the score is divided by sqrt(d_k) to prevent the scores from becoming too large

    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)
    
    # Apply softmax to get attention probabilities
    # 
    exp_scores = np.exp(scores)
    exp_scores = exp_scores - np.max(exp_scores, axis=-1, keepdims=True)  # for numerical stability
    attention_scores = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)  # (seq_len, seq_len)

    # update V 
    V = attention_scores @ V   # (seq_len, d_v）
    
    return attention_scores, V


text = 'I have two pets: one is a dog, the other is a cat.'
token_ids, tokenizer = tokenize(text)
attention_scores, V = get_self_attention_scores(token_ids, vocab_size=tokenizer.n_vocab, n_embedding=64, d_k=64, d_v=64)

print(f'text size: {len(text)}')
print(f'token size: {len(token_ids)}')
print(f'self attention score shape: {attention_scores.shape}')
print(f'matrix V shape: {V.shape}')

