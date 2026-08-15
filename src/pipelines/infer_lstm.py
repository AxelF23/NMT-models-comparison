import torch
import re


def translate_lstm(sentence, model, eng_vocab, ru_vocab, device, max_len=50, repetition_penalty=1.5):
    model.eval()

    # Preprocessing
    clean_sentence = re.sub(r'([.,!?"])', r' \1 ', sentence.lower())
    tokens = clean_sentence.strip().split()

    # Tokenization
    unk_idx = eng_vocab.word2index.get('UNK', 3)
    sos_idx = eng_vocab.word2index.get('SOS', 0)
    eos_idx = eng_vocab.word2index.get('EOS', 1)
    pad_idx = eng_vocab.word2index.get('PAD', 2)

    token_ids = [eng_vocab.word2index.get(tok, unk_idx) for tok in tokens]
    token_ids = [sos_idx] + token_ids + [eos_idx]

    # [seq_len, batch_size=1]
    src_tensor = torch.LongTensor(token_ids).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    input_token = torch.LongTensor([ru_vocab.word2index.get('SOS', 0)]).to(device)
    translated_words = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell)

        # repetition penalty
        if len(translated_words) > 0:
            last_word = translated_words[-1]
            last_word_idx = ru_vocab.word2index.get(last_word, None)

            if last_word_idx is not None and last_word_idx not in [unk_idx, pad_idx]:
                if output[0, last_word_idx] > 0:
                    output[0, last_word_idx] /= repetition_penalty
                else:
                    output[0, last_word_idx] *= repetition_penalty

        pred_token = output.argmax(1).item()

        if pred_token == eos_idx:
            break

        word = ru_vocab.index2word.get(pred_token, 'UNK')
        translated_words.append(word)

        input_token = torch.LongTensor([pred_token]).to(device)

    return " ".join(translated_words)