import os
import glob
from data_tools import text_length, preprocess_text, filter_short_texts
from sklearn.model_selection import train_test_split
import re
import pandas as pd


def read_texts_with_labels(folder_path, df_labels):
    texts = []
    labels = []
    file_ids = []

    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))
    for file_path in file_paths:
        file_name = os.path.basename(file_path)  # 只要文件名，不含路径
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 用文件名去df_labels匹配对应label
        file_id = os.path.splitext(file_name)[0]
        matched_rows = df_labels[df_labels['file_id'] == file_id]
        if not matched_rows.empty:
            label = matched_rows.iloc[0]['label']  # 取第一个匹配的label
        else:
            label = None  # 或者设定默认值，比如 'unknown'

        file_ids.append(file_name)
        texts.append(preprocess_text(content))
        labels.append(label)

    # 转成DataFrame方便后续处理
    df = pd.DataFrame({
        'file_id': file_ids,
        'text': texts,
        'label': labels
    })
    return df


def read_white_files(save_csv=False):
    root_dir = "./dataset/white/all_files"
    label = pd.read_csv('./dataset/white/annotations_metadata.csv')

    data_all = read_texts_with_labels(root_dir, label)
    data_all = data_all[(data_all['label'] == 'hate') | (data_all['label'] == 'noHate')]

    # Preprocess and filter short texts
    duplicates = data_all['text'].duplicated(keep=False)
    data_unique = data_all[~duplicates].copy()
    data_unique['label'] = data_unique['label'].isin(['hate']).astype(int)
    text_length(data_unique['text'].tolist())

    texts_fil, labels_fil = filter_short_texts(data_unique['text'].tolist(), data_unique['label'].tolist(),
                                               min_length=2)

    # Split train / val / test
    train_texts, tmp_texts, train_labels, tmp_labels = train_test_split(
        texts_fil, labels_fil, test_size=0.2, random_state=42, stratify=labels_fil)

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        tmp_texts, tmp_labels, test_size=0.5, random_state=42, stratify=tmp_labels)

    train_counts = pd.Series(train_labels).value_counts()
    val_counts = pd.Series(val_labels).value_counts()
    test_counts = pd.Series(test_labels).value_counts()

    # Optionally save to CSV
    if save_csv:
        train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
        val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
        test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

        train_df.to_csv('./dataset/white/train.csv', index=True, index_label="id")
        val_df.to_csv('./dataset/white/val.csv', index=True, index_label="id")
        test_df.to_csv('./dataset/white/test.csv', index=True, index_label="id")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

# read_white_files(save_csv=True)
