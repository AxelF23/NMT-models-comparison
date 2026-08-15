import argparse
import pickle
import torch
import pandas as pd
from tqdm.auto import tqdm
from nltk.translate.bleu_score import corpus_bleu

# import model architectures
from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq
from src.models.lstm_with_luong_attention import Encoder_att, Decoder_att, Seq2Seq_att
from src.models.transformer import Transformer

# import inference helpers
from src.pipelines.infer_lstm import translate_lstm
from src.pipelines.infer_lstm_attention import translate_lstm_attention
from src.pipelines.infer_transformer import translate_transformer


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model on test set")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lstm', 'lstm_attention', 'transformer'],
        help='Model architecture to evaluate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='data/processed/test.csv',
        help='Path to the test dataset'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    # load test dataset and saved vocabularies
    print(f"loading test data from {args.test_data}")
    test_df = pd.read_csv(args.test_data)

    print("loading vocabularies...")
    with open("data/processed/eng_vocab.pkl", "rb") as f:
        eng_vocab = pickle.load(f)
    with open("data/processed/ru_vocab.pkl", "rb") as f:
        ru_vocab = pickle.load(f)

    # initialize selected model and load weights
    print(f"loading {args.model} weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.model == 'lstm':
        enc = Encoder(eng_vocab.n_words, 256, 512)
        dec = Decoder(ru_vocab.n_words, 512, 256)
        model = Seq2Seq(enc, dec, device).to(device)
        model.load_state_dict(checkpoint['model_state'])
        infer_func = translate_lstm

    elif args.model == 'lstm_attention':
        enc = Encoder_att(eng_vocab.n_words, 256, 512)
        dec = Decoder_att(ru_vocab.n_words, 512, 256)
        model = Seq2Seq_att(enc, dec, device).to(device)
        model.load_state_dict(checkpoint['model_state'])
        infer_func = translate_lstm_attention

    elif args.model == 'transformer':
        model = Transformer(
            src_vocab_size=eng_vocab.n_words,
            trg_vocab_size=ru_vocab.n_words,
            d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.2,
            src_pad_idx=2, trg_pad_idx=2
        ).to(device)

        state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model_state'
        model.load_state_dict(checkpoint[state_key])
        infer_func = translate_transformer

    # run translations on test samples
    print(f"starting evaluation on {len(test_df)} samples...")
    references = []
    hypotheses = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="translating"):
        src_text = str(row['english'])
        gt_text = str(row['non_english'])

        # generate model prediction
        pred_text = infer_func(src_text, model, eng_vocab, ru_vocab, device)

        # tokenize ground truth for nltk bleu
        ref_tokens = gt_text.lower().split()
        references.append([ref_tokens])

        # tokenize prediction
        hyp_tokens = pred_text.lower().split()
        hypotheses.append(hyp_tokens)

    # calculate total corpus bleu
    bleu_score = corpus_bleu(references, hypotheses) * 100

    print("\nevaluation results:")
    print(f"model: {args.model}")
    print(f"test samples: {len(test_df)}")
    print(f"corpus BLEU score: {bleu_score:.2f}")


if __name__ == '__main__':
    main()