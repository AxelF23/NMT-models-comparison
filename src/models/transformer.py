import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model (int): Hidden dimension size of input embeddings/states.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): Query tensor of shape [B, L_q, d_model]
            k (torch.Tensor): Key tensor of shape [B, L_k, d_model]
            v (torch.Tensor): Value tensor of shape [B, L_k, d_model]
            mask (torch.Tensor, optional): Mask tensor [B, 1, 1, L_k] or [B, 1, L_q, L_k]
        Returns:
            torch.Tensor: Attended output tensor of shape [B, L_q, d_model]
        """

        B = q.shape[0]
        L_q = q.shape[1]
        L_k = k.shape[1]

        d_k = self.d_model // self.num_heads


        Q = self.W_q(q)  # [B, L_q, d_model]
        K = self.W_k(k)  #[B, L_k, d_model]
        V = self.W_v(v)  # [B, L_k, d_model]

        # Reshape to separate the embedding dimension into multiple attention heads
        Q = Q.reshape(B, L_q, self.num_heads, d_k)  # [B, L_q, num_heads, d_k]
        K = K.reshape(B, L_k, self.num_heads, d_k)  # [B, L_q, num_heads, d_k]
        V = V.reshape(B, L_k, self.num_heads, d_k)  # [B, L_q, num_heads, d_k]


        Q = Q.transpose(1, 2)  # [B, num_heads, L_q, d_k]
        K = K.transpose(1, 2)  # [B, num_heads, L_k, d_k]
        V = V.transpose(1, 2)  # [B, num_heads, L_k, d_k]

        K_T = K.transpose(2, 3) # [B, num_heads, d_k, L_k]
        Scores = (Q @ K_T) / math.sqrt(d_k) # [B, num_heads, L_q, L_k]

        # Causal Masking
        if mask is not None:
            Scores = Scores.masked_fill(mask == 0, torch.finfo(Scores.dtype).min)


        A = F.softmax(Scores, dim=-1) # [B, num_heads, L_q, L_k]
        output = A @ V  # [B, num_heads, L_q, d_k]

        output = output.transpose(1, 2)  # [B, L_q, num_heads, d_k]
        output = output.flatten(start_dim=2)  # [B, L_q, d_model]


        O = self.W_o(output)  # [B, L_q, d_model]
        return O

class FeedForward(nn.Module):
    """
     Feed-Forward Network.
        Args:
            d_model (int): Input and output dimensional size
            d_ff (int): Hidden layer dimensionality
            dropout (float): Dropout probability
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_1 = nn.Linear(d_model, d_ff, bias = True)
        self.dropout = nn.Dropout(p = dropout)
        self.W_2 = nn.Linear(d_ff, d_model, bias = True)
    def forward(self, x):
        # x: [B, L, d_model]
        x = self.W_1(x) # [B, L, d_ff]
        x = F.relu(x)
        x = self.dropout(x)
        x = self.W_2(x)   # [B, L, d_model]
        return x

class EncoderLayer(nn.Module):
    """
    Single Encoder Layer consisting of Self-Attention and FFN with residual connections

    Args:
        d_model (int): Hidden feature dimension
        d_ff (int): Inner dimension of FFN
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, d_model, d_ff, num_heads, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.FFN = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask):
        # x: [B, L_src, d_model]
        # mask: [B, 1, 1, L_src]
        x_norm = self.norm1(x)  # [B, L_src, d_model]
        attn_out = self.self_attn(q=x_norm, k=x_norm, v=x_norm, mask=mask) # [B, L_src, d_model]
        x = x + self.dropout1(attn_out) # [B, L_src, d_model]
        x_norm = self.norm2(x) # [B, L_src, d_model]
        ffn_out = self.FFN(x_norm) # [B, L_src, d_model]
        x = x + self.dropout2(ffn_out) # [B, L_src, d_model]
        return x

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    Args:
        d_model (int): Hidden dimensionality
        dropout (float): Dropout rate
        max_len (int): Maximum sequence length supported
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, L, d_model]
        x = x + self.pe[:, :x.size(1)]  # [B, L, d_model]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, num_heads, dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff,num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
    def forward(self, x, mask = None):
        # x: [B, L_src]
        x = self.embedding(x) * math.sqrt(self.d_model) # [B, L_src, d_model]
        x = self.pos_encoding(x)    # [B, L_src, d_model]
        for layer in self.layers:
            x = layer(x, mask)  # [B, L_src, d_model]
        return self.norm(x)    # [B, L_src, d_model]


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer with Masked Self-Attention and Cross-Attention
    Args:
        d_model (int): Feature dimension size
        d_ff (int): Inner dimension of FFN
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask = None, trg_mask = None):
        # x: [B, L_trg, d_model]
        # enc_out: [B, L_src, d_model]
        x_norm = self.norm1(x)
        attn_out = self.self_attn(q=x_norm, k=x_norm, v=x_norm, mask=trg_mask) # [B, L_trg, d_model]
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        cross_out = self.cross_attn(q=x_norm, k=enc_out, v=enc_out, mask=src_mask) # [B, L_trg, d_model]
        x = x + self.dropout2(cross_out)
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)  # [B, L_trg, d_model]
        x = x + self.dropout3(ffn_out)
        return x

class Decoder(nn.Module):
    """
    Args:
        vocab_size (int): Target vocabulary size
        d_model (int): Hidden dimension size
        d_ff (int): FFN hidden dimension size
        num_layers (int): Number of decoder layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        """
    def __init__(self, vocab_size, d_model, d_ff, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, enc_out, src_mask=None, trg_mask=None):

        # trg: [B, L_trg]
        # enc_out: [B, L_src, d_model]
        x = self.embedding(trg) # [B, L_trg, d_model]
        x = self.pos_encoding(x) # [B, L_trg, d_model]

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask) # [B, L_trg, d_model]

        x = self.norm(x)  # [B, L_trg, d_model]
        return self.fc_out(x)  # [B, L_trg, vocab_size]

class Transformer(nn.Module):
    """
    Full Transformer Seq2Seq Model.
    Args:
        src_vocab_size (int): Size of source vocabulary
        trg_vocab_size (int): Size of target vocabulary
        d_model (int): Hidden dimensionality (default: 256)
        num_layers (int): Number of Encoder and Decoder layers
        num_heads (int): Number of multi-head attention heads
        d_ff (int): Inner dimension of Feed-Forward network
        dropout (float): Dropout probability (default: 0.1)
        src_pad_idx (int): Padding index for source sequences
        trg_pad_idx (int): Padding index for target sequences
    """
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        d_model=256,
        num_layers=3,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        src_pad_idx=0,
        trg_pad_idx=0
    ):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_layers, num_heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, d_ff, num_layers, num_heads, dropout)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # src: [B, L_src]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_src]
        return src_mask

    def make_trg_mask(self, trg):
        # trg: [B, L_trg]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # [B, 1, 1, L_trg]

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        # src: [B, L_src]
        # trg: [B, L_trg]
        src_mask = self.make_src_mask(src)  # [B, 1, 1, L_src]
        trg_mask = self.make_trg_mask(trg)  # [B, 1, L_trg, L_trg]

        enc_out = self.encoder(src, src_mask) # [B, L_src, d_model]

        out = self.decoder(trg, enc_out, src_mask, trg_mask) # [B, L_trg, trg_vocab_size]

        return out