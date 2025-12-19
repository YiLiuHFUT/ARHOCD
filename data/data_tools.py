from nltk.tokenize import word_tokenize
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re


def preprocess_text(text: str) -> str:
    "Kumbam P R, Syed S U, Thamminedi P, et al. Exploiting Explainability to Design Adversarial Attacks and Evaluate Attack Resilience in Hate-Speech Detection Models[C]//Proceedings of the International AAAI Conference on Web and Social Media. 2025, 19: 1038-1050."
    # lower
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove @user and #hashtag
    text = re.sub(r"@\w+|#\w+", "", text)
    # normalize space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_length(text_data):
    """
        Args:
        text_data (List[str])
    """
    word_counts = [len(word_tokenize(text)) for text in text_data]
    max_word_count = max(word_counts)
    min_word_count = min(word_counts)
    avg_word_count = sum(word_counts) / len(word_counts)
    print(f'max: {max_word_count}, min: {min_word_count}, avg: {avg_word_count}')
    return word_counts


def filter_short_texts(text_list, label_list, min_length=5):
    """
        Args:
        text_list (List[str])
        label_list (List[int])
        min_length (int)
    """
    filtered_texts = []
    filtered_labels = []

    for text, label in zip(text_list, label_list):
        if len(word_tokenize(text)) >= min_length:
            filtered_texts.append(text)
            filtered_labels.append(label)

    return filtered_texts, filtered_labels
