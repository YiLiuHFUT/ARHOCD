import os
from collections import Counter
import pandas as pd
import argparse
import numpy as np
from torch.utils.data import DataLoader
from beifen.test_model import test
from tools import cal_metrics, load_model, MyDataset
from tqdm import tqdm
from get_attack_file import attack_file_combine_for_single_model, get_new_attack_file, get_benign_data
from itertools import product
from joblib import Parallel, delayed
from ensemble_data import PredictionDataset, test_aggre, train_aggre
from ensemble_bayesian import test_aggre_bayesian, calculate_weights, train_aggre_bayesian
from aggregation_model import CrossAttentionWeightGenerator, MLPWeightGenerator
import torch
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_params(all_predictions, labels, alpha, beta):
    weight_matrix = calculate_weights(all_predictions, alpha=alpha, belta=beta)
    aggregated_prob = np.sum(all_predictions * weight_matrix[:, :, :, np.newaxis], axis=(1, 2))
    aggregated_acc, aggregated_p2, aggregated_r2, aggregated_f2, aggregated_auc, aggregated_acc0, aggregated_acc1 = cal_metrics(
        labels, (aggregated_prob[:, 1] >= 0.5).astype(int), aggregated_prob, return_per_class_acc=True)
    return aggregated_acc, aggregated_p2, aggregated_r2, aggregated_f2, aggregated_auc, aggregated_acc0, aggregated_acc1, (
        alpha, beta)


def collect_benign_predictions(data, models, tokenizers, n_variants, n_models, n_classes, random_state):
    """
        Args:
        data (pd.DataFrame): Input dataset containing columns ['id', 'text', 'label'].
        models (dict): A dictionary of model_name → model_object.
        tokenizers (dict): A dictionary of model_name → tokenizer.
        n_variants (int): Number of variants to collect per sample.
        n_models (int): Number of models in the ensemble.
        n_classes (int): Number of output classes.
        random_state (int): Random seed for sampling.

    Returns:
        preds (np.ndarray): Prediction tensor of shape
            [num_samples, n_variants, n_models, n_classes].
        labels_out (List[int]): Ground-truth labels for each sample.
    """
    total_samples = len(data['id'].unique())
    # print(total_samples)
    preds = np.zeros((total_samples, n_variants, n_models, n_classes))
    labels_out = []

    texts_flat, labels_flat, index_map = [], [], []
    for sample_idx, (name, group) in enumerate(data.groupby("id")):
        first_row = group.iloc[[0]]
        remaining_rows = group.iloc[1:]
        n_random = min(n_variants - 1, len(remaining_rows))
        sampled_remaining = (
            remaining_rows.sample(n=n_random, random_state=random_state)
            if n_random > 0 else pd.DataFrame()
        )
        sampled_df = pd.concat([first_row, sampled_remaining], ignore_index=True)

        texts = sampled_df['text'].tolist()
        labels = [first_row['label'].iloc[0]] * len(texts)
        labels_out.append(first_row['label'].iloc[0])

        # Repeat if the number of samples is insufficient
        if len(texts) < n_variants:
            times = (n_variants + len(texts) - 1) // len(texts)
            texts = (texts * times)[:n_variants]
            labels = (labels * times)[:n_variants]

        # Store texts for batched prediction later
        for variant_idx, (t, l) in enumerate(zip(texts, labels)):
            texts_flat.append(t)
            labels_flat.append(l)
            index_map.append((sample_idx, variant_idx))

    # Batched prediction for each model
    for model_idx, model_name_eval in enumerate(models):
        dataset = MyDataset(texts_flat, labels_flat, tokenizers[model_name_eval])
        loader = DataLoader(dataset, batch_size=32)
        probs_all = test(models[model_name_eval], loader)  # [N_total, n_classes]

        for i, (sample_idx, variant_idx) in enumerate(index_map):
            preds[sample_idx, variant_idx, model_idx, :] = probs_all[i]
    return preds, labels_out


def collect_predictions(data, models, tokenizers, n_variants, n_models, n_classes, random_state):
    """
        Args:
        data (pd.DataFrame): DataFrame containing
            ['original_text', 'perturbed_text', 'ground_truth_output', 'source'].
        models (dict): A dictionary of model_name → model_object.
        tokenizers (dict): A dictionary of model_name → tokenizer.
        n_variants (int): Number of adversarial variants per sample.
        n_models (int): Number of models in the ensemble.
        n_classes (int): Number of output classes.
        random_state (int): Seed for random selection.

    Returns:
        preds (np.ndarray): Prediction tensor of shape
            [num_samples, n_variants, n_models, n_classes].
        labels_out (List[int]): Ground-truth labels for each sample.
    """
    rng = np.random.RandomState(random_state)
    total_samples = len(data['original_text'].unique())
    # print(total_samples)
    preds = np.zeros((total_samples, n_variants, n_models, n_classes))
    labels_out = []

    texts_flat, labels_flat, index_map = [], [], []
    for sample_idx, (name, group) in enumerate(data.groupby("original_text")):
        # Randomly choose one attack generation model
        available_models = group['source'].unique()
        random_model = rng.choice(available_models)
        seleted_rows = group[group["source"] == random_model]

        if len(seleted_rows) == 0:
            continue

        # Select the original perturbed text + additional variants
        first_row = seleted_rows.iloc[[0]]
        remaining_rows = seleted_rows.iloc[1:]
        n_random = min(n_variants - 1, len(remaining_rows))
        sampled_remaining = (
            remaining_rows.sample(n=n_random, random_state=random_state)
            if n_random > 0 else pd.DataFrame()
        )
        sampled_df = pd.concat([first_row, sampled_remaining], ignore_index=True)

        texts = sampled_df['perturbed_text'].tolist()
        labels = [seleted_rows['ground_truth_output'].iloc[0]] * len(texts)
        labels_out.append(seleted_rows['ground_truth_output'].iloc[0])

        # Expand variants to match n_variants if necessary
        if len(texts) < n_variants:
            times = (n_variants + len(texts) - 1) // len(texts)
            texts = (texts * times)[:n_variants]
            labels = (labels * times)[:n_variants]

        # Store texts and indices for batch inference later
        for variant_idx, (t, l) in enumerate(zip(texts, labels)):
            texts_flat.append(t)
            labels_flat.append(l)
            index_map.append((sample_idx, variant_idx))

    # Batched prediction across all models
    for model_idx, model_name_eval in enumerate(models):
        dataset = MyDataset(texts_flat, labels_flat, tokenizers[model_name_eval])
        loader = DataLoader(dataset, batch_size=128)
        probs_all = test(models[model_name_eval], loader)  # [N_total, n_classes]

        for i, (sample_idx, variant_idx) in enumerate(index_map):
            preds[sample_idx, variant_idx, model_idx, :] = probs_all[i]
    return preds, labels_out


def run_once(models, tokenizers, val_data, val_data_benign, test_data, random_state=42, benign=False, train_agg=False):
    """
    models : dict
        Mapping of model names to model instances.
    tokenizers : dict
        Corresponding tokenizers for each model.
    val_data : pd.DataFrame
        Validation dataset containing adversarial samples.
    val_data_benign : pd.DataFrame
        Validation dataset containing benign samples.
    test_data : pd.DataFrame
        Testing data (adversarial or benign, depending on setting).
    random_state : int, optional
        Random seed for reproducibility.
    benign : bool, optional
        If True, evaluate on benign data using `collect_benign_predictions`.
    train_agg : bool, optional
        If True, train aggregation models using validation data.

    Returns
    -------
    tuple
        Performance metrics of all aggregation methods and their predictions.
    """
    n_variants = 6
    n_models = 5
    n_classes = 2

    if benign:
        test_predictions, test_labels = collect_benign_predictions(test_data, models, tokenizers, n_variants, n_models,
                                                                   n_classes, random_state)
    else:
        test_predictions, test_labels = collect_predictions(test_data, models, tokenizers, n_variants, n_models,
                                                            n_classes, random_state)

    test_dataset = PredictionDataset(test_predictions, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #  Create output directories
    data_save_path = f'./agg_model/data/'
    bayesian_save_path = f'./agg_model/bayesian/'
    os.makedirs(data_save_path, exist_ok=True)
    os.makedirs(bayesian_save_path, exist_ok=True)

    #  Train aggregation model
    if train_agg:
        val_predictions_adv, val_labels_adv = collect_predictions(val_data, models, tokenizers, n_variants, n_models,
                                                                  n_classes, random_state)
        val_predictions_benign, val_labels_benign = collect_benign_predictions(val_data_benign, models, tokenizers,
                                                                               n_variants, n_models, n_classes,
                                                                               random_state)
        # Merge validation sets
        val_predictions = np.concatenate([val_predictions_adv, val_predictions_benign], axis=0)
        val_labels = val_labels_adv + val_labels_benign

        # Class weights for imbalanced data
        counts = Counter(val_labels)
        total = sum(counts.values())
        weights = [total / counts[i] for i in sorted(counts.keys())]
        class_weights = torch.tensor(weights, dtype=torch.float).cuda()

        val_dataset = PredictionDataset(val_predictions, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # Train Data-driven MLP aggregator
        data_model = MLPWeightGenerator()
        train_aggre(data_model, val_loader, test_loader, path=data_save_path, dataset=args.dataset, lr=1e-6,
                    class_weights=class_weights)

        # Train Bayesian attention aggregator
        bayesian_model = CrossAttentionWeightGenerator()
        train_aggre_bayesian(bayesian_model, val_loader, test_loader, path=bayesian_save_path, alpha=1.6,
                             beta=0.8, dataset=args.dataset, lr=1e-3, lamda=0.001, class_weights=class_weights)

    # ----------------------------------------------------------------------
    #  Baseline 1: EEL (simple averaging)
    # ----------------------------------------------------------------------
    agg_prob = np.mean(test_predictions, axis=(1, 2))  # shape: (n_samples, n_classes)
    agg_acc, agg_p2, agg_r2, agg_f2, agg_auc, agg_acc_0, agg_acc_1 = cal_metrics(test_labels,
                                                                                 (agg_prob[:, 1] >= 0.5).astype(int),
                                                                                 agg_prob, return_per_class_acc=True)

    print('Average: ', agg_acc, agg_p2, agg_r2, agg_f2, agg_acc_0, agg_acc_1)
    EEL_metrics = (agg_acc, agg_p2, agg_r2, agg_f2, agg_acc_0, agg_acc_1)

    # ----------------------------------------------------------------------
    #  Baseline 2: CBWD (Confidence-Based Weighted Decision)
    # ----------------------------------------------------------------------
    prob_diff = np.abs(test_predictions[..., 0] - test_predictions[..., 1])
    weight_matrix_CBWD = prob_diff ** 3
    weights_sum = weight_matrix_CBWD.sum(axis=(1, 2), keepdims=True)
    normalized_weights = weight_matrix_CBWD / weights_sum
    CBWD_prob = np.sum(test_predictions * normalized_weights[:, :, :, np.newaxis], axis=(1, 2))
    CBWD_acc, CBWD_p2, CBWD_r2, CBWD_f2, CBWD_auc, CBWD_acc_0, CBWD_acc_1 = cal_metrics(test_labels,
                                                                                        (CBWD_prob[:, 1] >= 0.5).astype(
                                                                                            int), CBWD_prob,
                                                                                        return_per_class_acc=True)
    print('CBWD: ', CBWD_acc, CBWD_p2, CBWD_r2, CBWD_f2, CBWD_acc_0, CBWD_acc_1)
    CBWD_metrics = (CBWD_acc, CBWD_p2, CBWD_r2, CBWD_f2, CBWD_acc_0, CBWD_acc_1)


    # ----------------------------------------------------------------------
    #  Proposed Method 1: Prior-based Aggregation (Grid Search)
    # ----------------------------------------------------------------------
    alpha_list = np.arange(0.1, 3, 0.5)
    beta_list = np.arange(0.1, 3, 0.5)
    gamma = [0.5]

    param_combinations = list(product(alpha_list, beta_list, gamma))
    results = Parallel(n_jobs=50)(
        delayed(evaluate_params)(val_predictions, val_labels, alpha, beta) for alpha, beta in
        tqdm(param_combinations, desc="Grid Search"))
    best_score = -1
    for prior_acc, prior_p2, prior_r2, prior_f2, prior_auc, prior_acc_0, prior_acc_1, params in results:
        if prior_f2 >= best_score:
            best_score = prior_f2
            best_params = params
    best_alpha, best_beta = map(float, best_params)
    prior_acc, prior_p2, prior_r2, prior_f2, prior_auc, prior_acc_0, prior_acc_1, *_ = evaluate_params(test_predictions,
                                                                                                       test_labels,
                                                                                                       best_alpha,
                                                                                                       best_beta)
    print("Prior: ", prior_acc, prior_p2, prior_r2, prior_f2, prior_acc_0, prior_acc_1)
    print("Best params:", best_params)
    prior_metrics = (prior_acc, prior_p2, prior_r2, prior_f2, prior_acc_0, prior_acc_1)

    # ----------------------------------------------------------------------
    #  Proposed Method 2: Data-driven Aggregation (MLP)
    # ----------------------------------------------------------------------
    data_model = MLPWeightGenerator()
    data_model.load_state_dict(torch.load(os.path.join(data_save_path, f'{args.dataset}.pt')))
    _, data_acc, data_p2, data_r2, data_f2, data_acc0, data_acc1 = test_aggre(data_model, test_loader,
                                                                              return_per_class_acc=True)
    data_metrics = (data_acc, data_p2, data_r2, data_f2, data_acc0, data_acc1)
    print("data_metrics", data_acc, data_p2, data_r2, data_f2, data_acc0, data_acc1)

    # ----------------------------------------------------------------------
    #  Proposed Method 3: Bayesian Aggregation (MC sampling & Expectation)
    # ----------------------------------------------------------------------
    bayesian_model = CrossAttentionWeightGenerator()
    bayesian_model.load_state_dict(torch.load(os.path.join(bayesian_save_path, f'{args.dataset}.pt')))
    _, MC_acc, MC_p2, MC_r2, MC_f2, MC_acc0, MC_acc1 = test_aggre_bayesian(bayesian_model, test_loader, MC=True,
                                                                           return_per_class_acc=True)
    _, Exp_acc, Exp_p2, Exp_r2, Exp_f2, Exp_acc0, Exp_acc1 = test_aggre_bayesian(bayesian_model, test_loader, MC=False,
                                                                                 return_per_class_acc=True)
    MC_metrics = (MC_acc, MC_p2, MC_r2, MC_f2, MC_acc0, MC_acc1)
    Exp_metrics = (Exp_acc, Exp_p2, Exp_r2, Exp_f2, Exp_acc0, Exp_acc1)
    print("Bayesian (MC)", MC_acc, MC_p2, MC_r2, MC_f2, MC_acc0, MC_acc1)
    print("Bayesian (Expectation)", Exp_acc, Exp_p2, Exp_r2, Exp_f2, Exp_acc0, Exp_acc1)
    return EEL_metrics, CBWD_metrics, prior_metrics, data_metrics, MC_metrics, Exp_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='white')
    parser.add_argument('--sample_path', type=str, default='white/ST_ours')
    parser.add_argument('--model_path', type=str, default='STmodel')
    args = parser.parse_args()

    set_seed(42)

    # ===== Load base models and tokenizers =====
    model_names = ['bert', 'distilbert', 'roberta', 'xlnet', 'albert']
    models, tokenizers = {}, {}

    for name in model_names:
        model, tokenizer = load_model(f'./{args.model_path}/{args.dataset}/{name}', model_max_length=256)
        models[name] = model
        tokenizers[name] = tokenizer

    # ===== Load validation adversarial and benign data =====
    val_attack_dfs = [
        attack_file_combine_for_single_model(args.sample_path, name, type='val') for name in model_names
    ]
    val_data_adv = pd.concat(val_attack_dfs, ignore_index=True)
    val_data_adv = val_data_adv.sample(frac=0.4, random_state=42).reset_index(drop=True)

    val_data_benign = get_benign_data(sample_path=f'{args.dataset}/ST_ours', type='val')
    val_data_benign = val_data_benign.sample(frac=0.4, random_state=42).reset_index(drop=True)

    # ===== Define experimental configurations =====
    configs = [
        {'attack_type': 'attack', 'cal_asr': False},
        {'attack_type': 'attack', 'cal_asr': True},
        {'attack_type': 'benign', 'cal_asr': False},
        {'attack_type': 'new_attack', 'cal_asr': False},
        {'attack_type': 'new_attack', 'cal_asr': True},
    ]

    # ===== Run experiments =====
    for cfg in configs:
        if cfg["attack_type"] == 'attack':
            benign = False
            test_dfs = [
                attack_file_combine_for_single_model(args.sample_path, name, type='test') for name in model_names
            ]
            test_data = pd.concat(test_dfs, ignore_index=True)

            train_agg = not cfg["cal_asr"]

            if cfg["cal_asr"]:
                # Exclude skipped samples for ASR calculation
                test_data = test_data[test_data['result_type'] != 'Skipped']

        # ===== New attack (ExplainDrive) =====
        elif cfg["attack_type"] == 'new_attack':
            benign = False
            train_agg = False
            test_dfs = [
                get_new_attack_file(args.sample_path, name, type='test') for name in model_names
            ]
            test_data = pd.concat(test_dfs, ignore_index=True)

            if cfg["cal_asr"]:
                test_data = test_data[test_data["result_type"] != "Skipped"]

        elif cfg["attack_type"] == 'benign':
            benign = True
            train_agg = False
            test_data = get_benign_data(sample_path=f'{args.dataset}/ST_ours', type='test')

        EEL_metrics, CBWD_metrics, prior_metrics, data_metrics, MC_metrics, Exp_metrics \
            = run_once(models, tokenizers, val_data=val_data_adv, val_data_benign=val_data_benign, test_data=test_data,
                       random_state=42, benign=benign, train_agg=train_agg)
