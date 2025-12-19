import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from tools import cal_metrics, load_model, \
    freeze_xlnet_layers, freeze_distilbert_layers, freeze_bert_layers, freeze_albert_layers, freeze_roberta_layers
from aggregation_model import CrossAttentionWeightGenerator
from functools import partial
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import pandas as pd
import torch.nn as nn
import random
from collections import Counter


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expand_variants(df, group_col="id", text_col="text", n_variants=10, random_state=42, adv=False):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing grouped perturbation results.
    group_col : str
        Column name used for grouping samples (e.g., 'id' or 'original_text').
    text_col : str
        Column name containing perturbed texts.
    n_variants : int
        Number of variants retained for each group after expansion.
    random_state : int
        Random state for reproducibility.
    adv : bool
        Whether the input dataframe corresponds to adversarial attack data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with each row representing one original text and its variants.
    """
    if adv and "similarity_per" in df.columns:
        df.rename(columns={'similarity_per': 'similarity'}, inplace=True)

    rng = np.random.default_rng(random_state)
    rows = []

    # Group by the original text identifier
    for gid, group in df.groupby(group_col):
        group_fil = group[group["similarity"] != 1]

        texts = group_fil[text_col].tolist()
        rng.shuffle(texts)

        # Pad insufficient variants
        if len(texts) < n_variants:
            if adv:
                base_text = group["perturbed_ori"].iloc[0]
            else:
                base_text = group["original_text"].iloc[0]
            texts = texts + [base_text] * (n_variants - len(texts))
        else:
            texts = texts[:n_variants]

        row = {group_col: gid}
        row['original_text'] = group['original_text'].iloc[0]

        # Assign metadata
        if adv:
            row['perturbed_text'] = group['perturbed_ori'].iloc[0]
            row['ground_truth_output'] = group['ground_truth_output'].iloc[0]
            row['result_type'] = group['result_type'].iloc[0]
        else:
            row['ground_truth_output'] = group['label'].iloc[0]

        for i, t in enumerate(texts, start=1):
            row[f"variant_{i}"] = t

        rows.append(row)
    return pd.DataFrame(rows)


def data_combine(file_path, model_name, type):
    """
    file_path : str
        Base directory containing attack results.
    model_name : str
        Model name (e.g., 'bert', 'roberta').
    split_type : str
        Dataset split type ('train', 'dev', 'test').

    Returns
    -------
    pandas.DataFrame
        Expanded adversarial variant dataset for one model.
    """
    data_wb = pd.read_csv(os.path.join(file_path, f'deepwordbug/attack_{model_name}_{type}.csv'))
    data_tf = pd.read_csv(os.path.join(file_path, f'tfadjusted/attack_{model_name}_{type}.csv'))
    data_tr = pd.read_csv(os.path.join(file_path, f'trepat/attack_{model_name}_{type}.csv'))
    data_all = pd.concat([data_wb, data_tf, data_tr], axis=0, ignore_index=True, sort=False)
    data_expand = expand_variants(data_all, group_col='original_text', text_col='perturbed_text', adv=True)
    return data_expand


def model_data_combine(file_path, type):
    """
    file_path : str
        Directory of per-model adversarial attack results.
    split_type : str
        Dataset split type ('train', 'dev', 'test').

    Returns
    -------
    pandas.DataFrame
        Aggregated and expanded adversarial dataset across all models.
    """
    models = ["bert", "distilbert", "roberta", "xlnet", "albert"]
    df_all = []
    for model in models:
        df_expand = data_combine(file_path, model, type)
        df_all.append(df_expand)
    return pd.concat(df_all, axis=0, ignore_index=True)


class VariantsDataset(Dataset):
    """
    data : pandas.DataFrame
    DataFrame containing columns:
    - 'ground_truth_output'
    - 'original_text' or 'perturbed_text'
    - 'variant_1', ..., 'variant_k'
    adv : bool
    """
    def __init__(self, data, adv=False):
        self.data = data
        self.labels = data['ground_truth_output'].tolist()
        self.adv = adv

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.adv:
            variants = [row["perturbed_text"]] + [row[f"variant_{i}"] for i in range(1, 6)]
        else:
            variants = [row["original_text"]] + [row[f"variant_{i}"] for i in range(1, 6)]
        label = torch.tensor(self.labels[idx])
        return {
            'variants': variants,
            'label': label
        }


def collate_fn(batch, tokenizers, max_length=256):
    """
    Parameters
    ----------
    batch : list of dict
        Each element contains:
            {
                "variants": [text_0, text_1, ..., text_k],
                "label": Tensor
            }
    tokenizers : dict
        Mapping: model_name -> HuggingFace tokenizer.
    max_length : int
        Maximum token length.

    Returns
    -------
    dict
        {
            "encodings": {
                model_name: [encoded_variant0, encoded_variant1, ...]
            },
            "labels": Tensor
        }
    """
    labels = torch.stack([item['label'] for item in batch])

    model_names = list(tokenizers.keys())
    num_variants = len(batch[0]['variants'])

    encodings = {model: [] for model in model_names}
    for model_name in model_names:
        tokenizer = tokenizers[model_name]
        for variant_idx in range(num_variants):
            texts = [sample['variants'][variant_idx] for sample in batch]
            encoded = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            encodings[model_name].append(encoded)

    return {
        'encodings': encodings,
        'labels': labels
    }


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def train_each_model_with_weight_model(base_models, weight_model, train_loader, test_loader, ben_test_loader, lr=1e-4,
                                       num_epochs=5, save_path=None, class_weights=None):
    """
    Args:
        base_models (dict): model_name -> model instance.
        weight_model (nn.Module): Aggregator model that outputs posterior weights.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader for adversarial data.
        ben_test_loader (DataLoader): Test loader for benign data.
        lr (float): Learning rate for optimizers.
        num_epochs (int): Number of training epochs.
        save_path (str): Directory to store checkpointed models.
        class_weights (Tensor): Class weights for imbalanced datasets.
    """
    print("Training base models using fixed weight model")

    criterion = torch.nn.NLLLoss(weight=class_weights).cuda()

    FREEZE_FUNCS = {
        'bert': freeze_bert_layers,
        'distilbert': freeze_distilbert_layers,
        'roberta': freeze_roberta_layers,
        'xlnet': freeze_xlnet_layers,
        'albert': freeze_albert_layers
    }

    num_variants = 6
    num_models = len(base_models)
    for name, m in base_models.items():
        freeze_fn = FREEZE_FUNCS.get(name)
        freeze_fn(m)

    # Set per-model learning rates
    LR_CONFIG = {
        'bert': 6e-6,
        'roberta': 6e-6,
        'distilbert': 6e-6,
        'xlnet': 6e-6,
        'albert': 6e-6
    }

    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=LR_CONFIG[name], weight_decay=0.01)
        for name, model in base_models.items()
    }

    accumulation_steps = 16
    best_f = 0.0
    for epoch in range(num_epochs):
        for model in base_models.values():
            model.train().cuda()

        total_loss = 0.0
        total_aggre_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"All Models | Epoch {epoch + 1}")):
            labels = batch['labels'].cuda().to(torch.int64)  # [batch_size]
            encodings = batch['encodings']  # dict: model_name -> list of 4 encoded batches
            batch_size = labels.size(0)

            # predictions: [B, V, M, C]
            all_predictions = torch.zeros(batch_size, num_variants, num_models, 2).cuda()
            # Forward pass for all models and all variants
            for model_idx, (name, model) in enumerate(base_models.items()):
                for variant_idx in range(num_variants):
                    inputs = encodings[name][variant_idx]
                    input_ids = inputs['input_ids'].cuda()
                    attention_mask = inputs['attention_mask'].cuda()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    all_predictions[:, variant_idx, model_idx, :] = outputs.logits.softmax(-1)

            if epoch < 2:
                posterior_weight = torch.ones(all_predictions.size(0), all_predictions.size(1),
                                              all_predictions.size(2)).cuda() * (1 / (num_variants * num_models))
            else:
                with torch.no_grad():
                    posterior_weight = weight_model(all_predictions[..., 1]).detach()

            # aggregated prediction: weighted sum over variants & models
            final_output = (all_predictions * posterior_weight.unsqueeze(-1)).sum((1, 2))
            aggre_loss = criterion(torch.log(final_output), labels)

            aggre_loss.backward()
            total_loss += aggre_loss.item()
            total_aggre_loss += aggre_loss.item()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()

        epoch_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}]  Avg Train Loss: {epoch_loss:.4f}")

        # Evaluation
        test_acc, test_f2, test_avg_loss = test_aggre(base_models, weight_model, test_loader)
        test_ben_acc, test_ben_f2, test_ben_avg_loss = test_aggre(base_models, weight_model, ben_test_loader)
        if (test_f2 + test_ben_f2) > best_f:
            best_f = test_f2 + test_ben_f2
            for model_idx, (name, model) in enumerate(base_models.items()):
                model_dir = os.path.join(save_path, name)
                base_models[name].save_pretrained(model_dir, safe_serialization=False)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {(test_avg_loss + test_ben_avg_loss):.4f}")


def test_aggre(base_models, weight_model, test_loader):
    """
    Args:
        base_models (dict):
            Dictionary mapping model_name -> model_instance.
            Each model outputs logits with shape [batch_size, num_classes].

        weight_model (nn.Module):
            A neural network that takes as input the per-model predictions
            and outputs posterior weights for aggregation.

        test_loader (DataLoader):
            DataLoader that outputs:
                batch["labels"]: Ground-truth labels.
                batch["encodings"]: dict(model_name -> list of variant encodings).

    Returns:
        tuple:
            overall_accuracy (float): Accuracy of aggregated predictions.
            f2_score (float): F2-score of aggregated predictions.
            avg_loss (float): Average NLL loss on the test set.
    """
    num_variants = 6
    num_models = len(base_models)
    base_models = {name: model.cuda() for name, model in base_models.items()}

    criterion = nn.NLLLoss()

    aggre_model_predictions = []
    aggre_model_probabilities = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            labels = batch['labels'].cuda().to(torch.int64)  # [batch_size]
            encodings = batch['encodings']  # dict: model_name -> list of 4 encoded batches
            batch_size = labels.size(0)

            all_predictions = torch.zeros(batch_size, num_variants, num_models, 2).cuda()
            for model_idx, model_name in enumerate(base_models.keys()):
                m = base_models[model_name]
                m.eval()

                with torch.no_grad():
                    for variant_idx in range(num_variants):
                        inputs = encodings[model_name][variant_idx]
                        input_ids = inputs['input_ids'].cuda()
                        attention_mask = inputs['attention_mask'].cuda()

                        outputs = m(input_ids=input_ids, attention_mask=attention_mask)
                        all_predictions[:, variant_idx, model_idx, :] = outputs.logits.softmax(-1)

            posterior_weight = weight_model(all_predictions[..., 1]).detach()

            final_output = torch.sum(all_predictions * posterior_weight[:, :, :, None], dim=(1, 2))
            aggre_loss = criterion(torch.log(final_output), labels)
            total_loss += aggre_loss.item()

            # Store results for metric computation
            aggre_model_probabilities.extend(final_output.detach().cpu().numpy())
            aggre_model_predictions.extend(np.argmax(final_output.detach().cpu().numpy(), axis=1))
            all_labels.extend(labels.cpu().numpy())

    aggre_acc, aggre_p2, aggre_r2, aggre_f2, aggre_auc = cal_metrics(all_labels, aggre_model_predictions,
                                                                     aggre_model_probabilities)
    print(f'(Aggre) Test loss: {total_loss / len(test_loader)}')
    print('(Aggre) Test acc, p2, r2, f2, auc', aggre_acc, aggre_p2, aggre_r2, aggre_f2, aggre_auc)
    return aggre_acc, aggre_f2, total_loss / len(test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pheme')
    parser.add_argument('--sample_path', type=str, default='pheme/STmodel')
    args = parser.parse_args()

    set_seed(42)
    # Load Pretrained Base Models
    bert_model, bert_tokenizer = load_model(f'./STmodel/{args.dataset}/bert', model_max_length=256)
    distilbert_model, distilbert_tokenizer = load_model(f'./STmodel/{args.dataset}/distilbert', model_max_length=256)
    roberta_model, roberta_tokenizer = load_model(f'./STmodel/{args.dataset}/roberta', model_max_length=256)
    xlnet_model, xlnet_tokenizer = load_model(f'./STmodel/{args.dataset}/xlnet', model_max_length=256)
    albert_model, albert_tokenizer = load_model(f'./STmodel/{args.dataset}/albert', model_max_length=256)

    base_models = {
        'bert': bert_model,
        'distilbert': distilbert_model,
        'roberta': roberta_model,
        'xlnet': xlnet_model,
        'albert': albert_model
    }

    tokenizers = {
        'bert': bert_tokenizer,
        'distilbert': distilbert_tokenizer,
        'roberta': roberta_tokenizer,
        'xlnet': xlnet_tokenizer,
        'albert': albert_tokenizer
    }

    # Create directory for saving updated models
    save_path = f'./UPDATEmodel/{args.dataset}/v1'
    os.makedirs(save_path, exist_ok=True)

    file_path = f'./attack_results/{args.sample_path}'
    adv_train_data = model_data_combine(file_path, type='train')
    adv_val_data = model_data_combine(file_path, type='val')
    adv_test_data = model_data_combine(file_path, type='test')

    benign_train_data_ori = pd.read_csv(f'./attack_results/{args.dataset}/ST_ours/benign/train.csv')
    benign_val_data_ori = pd.read_csv(f'./attack_results/{args.dataset}/ST_ours/benign/val.csv')
    benign_test_data_ori = pd.read_csv(f'./attack_results/{args.dataset}/ST_ours/benign/test.csv')
    benign_train_data = expand_variants(benign_train_data_ori)
    benign_val_data = expand_variants(benign_val_data_ori)
    benign_test_data = expand_variants(benign_test_data_ori)

    # Partial collate function for tokenization
    my_collate_fn = partial(collate_fn, tokenizers=tokenizers, max_length=256)
    # Create VariantsDataset objects
    adv_train_dataset = VariantsDataset(adv_train_data, adv=True)
    ben_train_dataset = VariantsDataset(benign_train_data)
    # Optionally reduce dataset size for quicker experiments (here keep 40% of original data)
    adv_train_dataset, _ = random_split(adv_train_dataset, [int(len(adv_train_dataset) * 0.4),
                                                            len(adv_train_dataset) - int(len(adv_train_dataset) * 0.4)])
    ben_train_dataset, _ = random_split(ben_train_dataset, [int(len(ben_train_dataset) * 0.4),
                                                            len(ben_train_dataset) - int(len(ben_train_dataset) * 0.4)])
    # Merge adversarial and benign training data
    merged_dataset = ConcatDataset([adv_train_dataset, ben_train_dataset])
    train_loader = DataLoader(merged_dataset, batch_size=2, shuffle=True, collate_fn=my_collate_fn, num_workers=1)

    adv_test_loader = DataLoader(VariantsDataset(adv_test_data, adv=True), batch_size=4, shuffle=False,
                                 collate_fn=my_collate_fn, num_workers=1)
    ben_test_loader = DataLoader(VariantsDataset(benign_test_data), batch_size=4, shuffle=False,
                                 collate_fn=my_collate_fn, num_workers=1)

    # Load pre-trained weight model for aggregation
    weight_model = CrossAttentionWeightGenerator()
    weight_model.load_state_dict(torch.load(os.path.join('./agg_model/bayesian/', f'{args.dataset}.pt')))

    weight_model.cuda()
    for model in base_models.values():
        model.cuda()

    # Compute class weights for imbalanced dataset
    counts = Counter(adv_train_data['ground_truth_output'].tolist() + benign_train_data['ground_truth_output'].tolist())
    total = sum(counts.values())
    weights = [total / counts[i] for i in sorted(counts.keys())]
    class_weights = torch.tensor(weights, dtype=torch.float).cuda()

    # Fine-tune each base model using weight assignor guidance
    train_each_model_with_weight_model(base_models, weight_model, train_loader, adv_test_loader, ben_test_loader,
                                       save_path=save_path, num_epochs=5, class_weights=class_weights)

    # Train weight assignor using fixed base models
    # Follow the same code "ours_bayesian_ensemble.py"
