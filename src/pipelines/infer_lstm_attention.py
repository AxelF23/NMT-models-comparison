import torch


def translate_lstm_attention(sentence, model, src_vocab, trg_vocab, device, max_len=50):
    model.eval()

    # Tokenization and lowercase
    if isinstance(sentence, str):
        tokens = [word for word in sentence.lower().split()]
    else:
        tokens = [word.lower() for word in sentence]

    unk_idx = src_vocab.word2index.get('UNK', 3)
    sos_idx = src_vocab.word2index.get('SOS', 0)
    eos_idx = src_vocab.word2index.get('EOS', 1)

    src_indexes = [sos_idx] + [src_vocab.word2index.get(token, unk_idx) for token in tokens] + [eos_idx]

    # [seq_len, batch_size=1]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    trg_indexes = [trg_vocab.word2index.get('SOS', 0)]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, encoder_outputs, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab.word2index.get('EOS', 1):
            break

    trg_tokens = [trg_vocab.index2word.get(i, 'UNK') for i in trg_indexes]

    return " ".join(trg_tokens[1:-1])