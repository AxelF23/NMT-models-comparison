import os
import pickle
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.dataset import Vocab

RAW_DATA_PATH = "data/raw/tatoeba_en_ru.csv"
PROCESSED_DIR = "data/processed"


def prepare_data():
    """
    Data download and dictionary preparation
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # download dataset via Hugging Face Hub
    if not os.path.exists(RAW_DATA_PATH):
        print("Downloading dataset: sentence-transformers/parallel-sentences-tatoeba...")
        dataset = load_dataset("sentence-transformers/parallel-sentences-tatoeba", "en-ru")

        # Convert to pandas DataFrame and cache raw copy
        full_df = dataset['train'].to_pandas()
        full_df.to_csv(RAW_DATA_PATH, index=False)
    else:
        print("Loading raw data from local path:", RAW_DATA_PATH)
        full_df = pd.read_csv(RAW_DATA_PATH)

    train_full, test_df = train_test_split(full_df, test_size=0.05, random_state=42)

    train_df = train_full.iloc[:300000]
    val_df = test_df.iloc[:2000]


    print("Building English and Russian vocabularies")
    eng_vocab = Vocab('english')
    ru_vocab = Vocab('russian')

    for eng_sent, ru_sent in zip(train_df['english'], train_df['non_english']):
        eng_vocab.add_sentence(str(eng_sent))
        ru_vocab.add_sentence(str(ru_sent))

    print("Saving processed splits and vocabulary artifacts to:", PROCESSED_DIR)
    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    with open(f"{PROCESSED_DIR}/eng_vocab.pkl", "wb") as f:
        pickle.dump(eng_vocab, f)
    with open(f"{PROCESSED_DIR}/ru_vocab.pkl", "wb") as f:
        pickle.dump(ru_vocab, f)

    print(f"Data preparation complete. Vocab sizes -> EN: {eng_vocab.n_words}, RU: {ru_vocab.n_words}")


if __name__ == "__main__":
    prepare_data()