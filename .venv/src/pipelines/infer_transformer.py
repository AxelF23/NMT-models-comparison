import torch


def translate_transformer(sentence, model, eng_vocab, ru_vocab, device, max_len=50):
    model.eval()

    # Preprocessing
    tokens = sentence.lower().strip().split()

    unk_idx = eng_vocab.word2index.get('UNK', 3)
    sos_idx = eng_vocab.word2index.get('SOS', 0)
    eos_idx = eng_vocab.word2index.get('EOS', 1)

    src_indexes = [sos_idx] + [eng_vocab.word2index.get(word, unk_idx) for word in tokens] + [eos_idx]

    # [batch_size=1, seq_len]
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        enc_out = model.encoder(src_tensor, src_mask)

    trg_indexes = [ru_vocab.word2index.get('SOS', 0)]

    for i in range(max_len):
        trg_tensor = torch.tensor(trg_indexes, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            trg_mask = model.make_trg_mask(trg_tensor)
            output = model.decoder(trg_tensor, enc_out, src_mask, trg_mask)

        # Get logit for the last generated word
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == eos_idx:
            break

    translated_words = [ru_vocab.index2word.get(idx, 'UNK') for idx in trg_indexes]

    return " ".join(translated_words[1:-1])