import json
import pickle
from pprint import PrettyPrinter
from tqdm import tqdm
import networkx as nx
from data_tools import text_length, preprocess_text, filter_short_texts
from sklearn.model_selection import train_test_split
import pandas as pd


def read_pheme_files(save_csv=False):
    dir = "./dataset/PHEME/all-rnr-annotated-threads"
    events = [i for i in os.listdir(dir) if "." not in i]

    text_list = []
    label_list = []
    for event in events:
        # non-rumours
        non_rumour_list = [i for i in os.listdir(f"{dir}/{event}/non-rumours") if "." not in i]

        for root_id in tqdm(non_rumour_list,
                            desc=f"preprocessing {event} non-rumours:"):
            with open(f"{dir}/{event}/non-rumours/{root_id}/source-tweets/{root_id}.json", "r") as f:
                root_info: dict = json.load(f)
                text_list.append(root_info["text"])
                label_list.append(0)

        # rumours
        rumour_list = [i for i in os.listdir(f"{dir}/{event}/rumours") if "." not in i]

        for root_id in tqdm(rumour_list,
                            desc=f"preprocessing {event} rumours:"):
            with open(f"{dir}/{event}/rumours/{root_id}/source-tweets/{root_id}.json", "r") as f:
                root_info: dict = json.load(f)
                text_list.append(root_info["text"])
                label_list.append(1)

    # Show length statistics
    text_length(text_list)

    # Preprocess and filter short texts
    text_list = list(map(preprocess_text, text_list))
    texts_fil, labels_fil = filter_short_texts(text_list, label_list, min_length=1)
    count_0 = labels_fil.count(0)
    count_1 = labels_fil.count(1)

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

        train_df.to_csv('./dataset/PHEME/train.csv', index=True, index_label="id")
        val_df.to_csv('./dataset/PHEME/val.csv', index=True, index_label="id")
        test_df.to_csv('./dataset/PHEME/test.csv', index=True, index_label="id")

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

# read_pheme_files(save_csv=True)
