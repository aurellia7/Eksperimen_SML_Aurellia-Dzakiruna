import pandas as pd
import numpy as np
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '..', 'ulasan_tiktok_raw.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'ulasan_tiktok_preprocessing.csv')

def load_data(path):
    print(f"Loading data dari: {path}")
    df = pd.read_csv(path)
    print(f"Shape data: {df.shape}")
    return df

def assign_sentiment(rating):
    if rating <= 2:
        return 'Negatif'
    elif rating >= 4:
        return 'Positif'
    else:
        return 'Netral'

def label_engineering(df):
    print("Melakukan label engineering...")
    df['Sentimen'] = df['Rating'].apply(assign_sentiment)
    df = df[df['Sentimen'] != 'Netral'].copy()
    print(f"Shape setelah drop Netral: {df.shape}")
    return df

def select_columns(df):
    df = df[['Ulasan', 'Sentimen']].copy()
    df = df.dropna()
    print(f"Shape setelah seleksi kolom: {df.shape}")
    return df

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess(df):
    print("Melakukan text cleaning...")
    df['Ulasan_clean'] = df['Ulasan'].apply(clean_text)
    print("Melakukan stopword removal...")
    df['Ulasan_processed'] = df['Ulasan_clean'].apply(remove_stopwords)
    print("Preprocessing selesai!")
    return df

def encode_label(df):
    df['label'] = df['Sentimen'].map({'Positif': 1, 'Negatif': 0})
    return df

def finalize(df):
    df_final = df[['Ulasan_processed', 'label']].copy()
    df_final = df_final.rename(columns={'Ulasan_processed': 'teks'})
    df_final = df_final[df_final['teks'].str.strip() != '']
    df_final = df_final.dropna()
    print(f"Shape data final: {df_final.shape}")
    return df_final

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Data tersimpan di: {path}")

def main():
    print("Automate Preprocessing — Aurellia Dzakiruna")
    df = load_data(RAW_DATA_PATH)
    df = label_engineering(df)
    df = select_columns(df)
    df = preprocess(df)
    df = encode_label(df)
    df_final = finalize(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df_final['teks'],
        df_final['label'],
        test_size=0.2,
        random_state=42,
        stratify=df_final['label']
    )
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size : {X_test.shape[0]}")

    save_data(df_final, OUTPUT_PATH)
    print("\nPipeline preprocessing selesai!")
    return df_final

if __name__ == "__main__":
    main()