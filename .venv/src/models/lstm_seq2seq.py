class Encoder(nn.Module):
  """
    LSTM - based encoder
    Args:
        input_dim (int): Size of the source vocabulary.
        emb_dim (int): Dimension of input embedding vectors.
        hid_dim (int): Dimension of hidden state vectors in LSTM.
  """
  def __init__(self, input_dim, emb_dim, hid_dim):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers = 1)
  def forward(self, src):
    embedded = self.embedding(src)
    outputs, (hidden, cell) = self.lstm(embedded)
    return (hidden, cell)


class Decoder(nn.Module):
  """
  LSTM - based encoder
  Args:
        output_dim (int): Size of the target vocabulary.
        hid_dim (int): Dimension of hidden state vectors.
        emb_dim (int): Dimension of target embedding vectors.
  """
  def __init__(self, output_dim, hid_dim, emb_dim):
    super().__init__()
    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers= 1)
    self.fc_out = nn.Linear(hid_dim, output_dim)
  def forward(self, input, hidden, cell):
    input = input.unsqueeze(0)
    embedded = self.embedding(input)
    outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
    out = self.fc_out(hidden.squeeze(0))
    return out, hidden, cell



class Seq2Seq(nn.Module):
  """
  Orchestrator class
  """
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    # src: [src_len, batch_size]
    # trg: [trg_len, batch_size]
    hidden, cell = self.encoder(src)
    outputs = torch.zeros(trg.shape[0], trg.shape[1], self.decoder.fc_out.out_features).to(device)
    input = trg[0] # [batch_size]
    for t in range(1,trg.shape[0]):
      output, hidden, cell = self.decoder(input, hidden, cell)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1) # [batch_size]
      input = trg[t] if teacher_force else top1 # [batch_size]

    return outputs