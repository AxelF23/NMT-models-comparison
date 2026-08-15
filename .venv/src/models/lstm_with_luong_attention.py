class Encoder_att(nn.Module):
  """
  LSTM - based encoder
  Args:
        input_dim (int): Size of the source vocabulary.
        emb_dim (int): Dimension of the input embedding vectors.
        hid_dim (int): Dimension of the hidden state vectors in LSTM.
  """
  def __init__(self, input_dim, emb_dim, hid_dim):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers = 1)
  def forward(self, src):
    embedded = self.embedding(src)
    outputs, (hidden, cell) = self.lstm(embedded)
    return outputs, hidden, cell


class LuongAttention(nn.Module):
    """
    Multiplicative Luong attention mechanism
    Args:
        hid_dim (int): Dimension of hidden states for linear projection.
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim]
        s_i = hidden.permute(1, 0, 2)            # [batch_size, 1, hid_dim]
        H = encoder_outputs.permute(1, 0, 2)     # [batch_size, src_len, hid_dim]

        s_i_W = self.W(s_i)                      # [batch_size, 1, hid_dim]
        e_i = torch.bmm(s_i_W, H.permute(0, 2, 1)) # [batch_size, 1, src_len]

        alpha_i = F.softmax(e_i, dim=-1)
        a_i = torch.bmm(alpha_i, H)              # [batch_size, 1, hid_dim]
        a_i = a_i.permute(1, 0, 2)               # [1, batch_size, hid_dim]

        return a_i


class Decoder_att(nn.Module):
    """
    LSTM decoder
    Args:
        output_dim (int): Size of the target vocabulary.
        hid_dim (int): Dimension of the hidden state vectors.
        emb_dim (int): Dimension of the target embedding vectors.
    """
    def __init__(self, output_dim, hid_dim, emb_dim):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=1)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.attention = LuongAttention(hid_dim)
        self.W_c = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, input, encoder_outputs, hidden, cell):
        # input: [batch_size] -> [1, batch_size]
        input = input.unsqueeze(0)
        embedded = self.embedding(input)

        lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))


        context = self.attention(hidden, encoder_outputs)


        concat = torch.cat((lstm_out, context), dim=2)
        attentional_hidden = torch.tanh(self.W_c(concat))

        prediction = self.fc_out(attentional_hidden.squeeze(0))
        return prediction, hidden, cell