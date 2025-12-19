import pandas as pd
from data_tools import text_length, preprocess_text, filter_short_texts
from sklearn.model_selection import train_test_split


def read_ethos_files(save_csv=False):
    data = pd.read_csv('./dataset/ethos/Ethos_Dataset_Binary.csv', sep=';')
    data['isHate'] = (data['isHate'] >= 0.5).astype(int)

    val_counts = data['isHate'].value_counts()

    texts_all = data['comment'].tolist()

    text_length(data['comment'].tolist())

    labels_all = data['isHate'].tolist()

    # Preprocess and filter short texts
    data_all = list(map(preprocess_text, texts_all))
    texts_fil, labels_fil = filter_short_texts(data_all, labels_all,
                                               min_length=1)

    # Split train / val / test
    train_texts, tmp_texts, train_labels, tmp_labels = train_test_split(
        texts_fil, labels_fil, test_size=0.2, random_state=42, stratify=labels_fil)

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        tmp_texts, tmp_labels, test_size=0.5, random_state=42, stratify=tmp_labels)

    # Optionally save to CSV
    if save_csv:
        train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
        val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
        test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

        train_df.to_csv('./dataset/ethos/train.csv', index=True, index_label="id")
        val_df.to_csv('./dataset/ethos/val.csv', index=True, index_label="id")
        test_df.to_csv('./dataset/ethos/test.csv', index=True, index_label="id")

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

# read_ethos_files(save_csv=True)
