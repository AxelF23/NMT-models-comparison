import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.index2word = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.n_words = 4
        self.word2count = {}

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            self.add_word(word)


class TranslationDataset(Dataset):
    def __init__(self, df, src_vocab, trg_vocab):
        self.df = df
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.df.iloc[idx]['english']
        trg_text = self.df.iloc[idx]['non_english']

        src_indices = [0] + [self.src_vocab.word2index.get(w, 3) for w in src_text.lower().split()] + [1]
        trg_indices = [0] + [self.trg_vocab.word2index.get(w, 3) for w in trg_text.lower().split()] + [1]

        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'trg': torch.tensor(trg_indices, dtype=torch.long)
        }


def get_collate_fn(batch_first=True):
    """
    batch_first = True for transformer, False for lstm models
    """
    def collate_fn(batch):
        src_batch = [item['src'] for item in batch]
        trg_batch = [item['trg'] for item in batch]

        src_padded = pad_sequence(src_batch, batch_first=batch_first, padding_value=2)
        trg_padded = pad_sequence(trg_batch, batch_first=batch_first, padding_value=2)

        return src_padded, trg_padded

    return collate_fn