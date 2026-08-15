import argparse
import torch
import sys


from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq
from src.models.lstm_with_luong_attention import Encoder_att, Decoder_att, Seq2Seq_att
from src.models.transformer import Transformer

from src.pipelines.infer_lstm import translate_lstm
from src.pipelines.infer_lstm_attention import translate_lstm_attention
from src.pipelines.infer_transformer import translate_transformer


def main():
    parser = argparse.ArgumentParser(description="Interactive Translation Inference")
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'lstm_attention', 'transformer'],
                        help='Model architecture to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth file)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load vocabularies
    try:
        eng_vocab = torch.load('checkpoints/eng_vocab.pth')
        ru_vocab = torch.load('checkpoints/ru_vocab.pth')
        print("[INFO] Vocabularies loaded successfully.")
    except FileNotFoundError:
        print("[ERROR] Vocab files not found! Please ensure 'eng_vocab.pth' and 'ru_vocab.pth' exist.")
        sys.exit(1)

    # Load model
    print(f"[INFO] Loading {args.model} model from {args.checkpoint}...")
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

    print("[INFO] Model loaded successfully! Entering interactive mode.")
    print("-" * 50)
    print("Type 'quit' or 'exit' to stop the program.")
    print("-" * 50)

    # interactive loop
    while True:
        try:
            user_input = input("\nEN (Input): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_input.strip():
                continue

            translation = infer_func(user_input, model, eng_vocab, ru_vocab, device)
            print(f"RU (Pred) : {translation}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[ERROR] An error occurred during translation: {e}")


if __name__ == '__main__':
    main()