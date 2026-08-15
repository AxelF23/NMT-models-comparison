import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

# Internal module imports
from src.dataset import TranslationDataset, Vocab, get_collate_fn
from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq
from src.models.lstm_with_luong_attention import Encoder_att, Decoder_att, Seq2Seq_att
from src.models.transformer import Transformer

# Pipeline imports
from src.pipelines.train_lstm import run_lstm_pipeline
from src.pipelines.train_lstm_attention import run_lstm_attention_pipeline
from src.pipelines.train_transformer import run_transformer_pipeline


def main():
    parser = argparse.ArgumentParser(description="Translation Models Training Script")
    parser.add_argument(
        '--model',
        type=str,
        default='lstm_attention',
        choices=['lstm', 'lstm_attention', 'transformer'],
        help='Model architecture to train'
    )
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training for model: {args.model} on device: {device}")

    # --- 1. DATA AND VOCABULARY LOADING ---
    # Load dataset dataframes and vocabularies here
    # train_df = pd.read_csv('data/processed/train.csv')
    # val_df = pd.read_csv('data/processed/val.csv')
    # eng_vocab = ...
    # ru_vocab = ...

    # --- 2. MODEL SELECTION AND PIPELINE EXECUTION ---
    if args.model == 'lstm':
        collate_fn = get_collate_fn(batch_first=False)
        train_dataset = TranslationDataset(train_df, eng_vocab, ru_vocab)
        val_dataset = TranslationDataset(val_df, eng_vocab, ru_vocab)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

        enc = Encoder(eng_vocab.n_words, 256, 512)
        dec = Decoder(ru_vocab.n_words, 512, 256)
        model = Seq2Seq(enc, dec, device).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=ru_vocab.word2index['PAD'])

        run_lstm_pipeline(
            model, train_loader, val_loader, optimizer, criterion,
            ru_vocab, device, num_epochs=args.epochs
        )

    elif args.model == 'lstm_attention':
        collate_fn = get_collate_fn(batch_first=False)
        train_dataset = TranslationDataset(train_df, eng_vocab, ru_vocab)
        val_dataset = TranslationDataset(val_df, eng_vocab, ru_vocab)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

        enc_att = Encoder_att(eng_vocab.n_words, 256, 512)
        dec_att = Decoder_att(ru_vocab.n_words, 512, 256)
        model_att = Seq2Seq_att(enc_att, dec_att, device).to(device)

        optimizer_att = torch.optim.Adam(model_att.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=ru_vocab.word2index['PAD'])

        run_lstm_attention_pipeline(
            model_att, train_loader, val_loader, optimizer_att, criterion,
            ru_vocab, device, num_epochs=args.epochs
        )

    elif args.model == 'transformer':
        collate_fn = get_collate_fn(batch_first=True)
        train_dataset = TranslationDataset(train_df, eng_vocab, ru_vocab)
        val_dataset = TranslationDataset(val_df, eng_vocab, ru_vocab)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

        model = Transformer(
            src_vocab_size=eng_vocab.n_words,
            trg_vocab_size=ru_vocab.n_words,
            d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.2,
            src_pad_idx=2, trg_pad_idx=2
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=2e-4, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1
        )
        criterion = nn.CrossEntropyLoss(ignore_index=2)

        run_transformer_pipeline(
            model, train_loader, optimizer, scheduler, criterion,
            device, num_epochs=args.epochs
        )


if __name__ == '__main__':
    main()