import pandas as pd
import numpy as np


def attack_file_combine_for_single_model(file_path: str, model_name: str, type: str) -> pd.DataFrame:
    # Load attack results from different methods
    data_db = pd.read_csv(
        f'./attack_results/{file_path}/deepwordbug/attack_{model_name}_{type}.csv')
    data_tf = pd.read_csv(
        f'./attack_results/{file_path}/tfadjusted/attack_{model_name}_{type}.csv')
    data_tr = pd.read_csv(
        f'./attack_results/{file_path}/trepat/attack_{model_name}_{type}.csv')

    # Concatenate data frames from different methods
    merged_df = pd.concat([data_db, data_tf, data_tr], ignore_index=True)
    merged_df["source"] = model_name

    merged_df["ground_truth_output"] = merged_df["ground_truth_output"].astype(int)
    return merged_df


def get_new_attack_file(file_path: str, model_name: str, type: str) -> pd.DataFrame:
    data = pd.read_csv(
        f'./attack_results/{file_path}/explaindrive/attack_{model_name}_{type}.csv')

    data["source"] = model_name
    data["ground_truth_output"] = data["ground_truth_output"].astype(float).astype(int)
    return data


def get_benign_data(sample_path: str, type: str) -> pd.DataFrame:
    data = pd.read_csv(
        f'./attack_results/{sample_path}/benign/{type}.csv')
    data["label"] = data["label"].astype(int)
    return data
