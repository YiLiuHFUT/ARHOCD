import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from tools import cal_metrics, MyDataset, \
    freeze_albert_layers, freeze_distilbert_layers, freeze_bert_layers, freeze_xlnet_layers, freeze_roberta_layers
from collections import Counter
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          lr: float, num_epochs: int = 5, save_path: str = None, class_weights: torch.Tensor = None):
    print("Starting model training...")
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.95)

    best_val_loss = float('inf')
    best_f = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        predictions, all_labels, probs = [], [], []

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda().to(torch.int64)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.extend(preds)
            predictions.extend(np.argmax(preds, axis=1))
            all_labels.extend(labels.cpu().numpy())

        scheduler.step()
        epoch_acc, epoch_p2, epoch_r2, epoch_f2, epoch_auc = cal_metrics(all_labels, predictions, probs)
        epoch_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Train Accuracy: {epoch_acc:.4f}, Avg Train Loss: {epoch_loss:.4f}")
        val_acc, val_f, val_avg_loss, _, _, _ = test(model, val_loader)
        if val_f > best_f or (val_f == best_f and val_avg_loss < best_val_loss):
            best_f = val_f
            best_val_loss = val_avg_loss
            model.save_pretrained(save_path, safe_serialization=False)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {val_avg_loss:.4f}")


def test(model: torch.nn.Module, test_loader: DataLoader):
    print("Evaluating model...")
    model = model.cuda()
    model.eval()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    predictions, all_labels, probs = [], [], []

    test_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda().to(torch.int64)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            test_loss += loss.item()

            preds = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.extend(preds)
            predictions.extend(np.argmax(preds, axis=1))
            all_labels.extend(labels.cpu().numpy())

    acc, p2, r2, f2, auc = cal_metrics(all_labels, predictions, probs)
    avg_loss = test_loss / len(test_loader)
    print(f'Test Accuracy: {acc:.4f}, Avg Test loss: {avg_loss:.4f}')
    return acc, f2, avg_loss, predictions, probs, all_labels


def get_data(dataset: str):
    if dataset == 'pheme':
        file_dir = "PHEME"
    else:
        file_dir = dataset

    train_data = pd.read_csv(f'./dataset/{file_dir}/train.csv')
    train_texts = train_data['text'].tolist()
    train_labels = train_data['label'].tolist()

    val_data = pd.read_csv(f'./dataset/{file_dir}/val.csv')
    val_texts = val_data['text'].tolist()
    val_labels = val_data['label'].tolist()

    test_data = pd.read_csv(f'./dataset/{file_dir}/test.csv')
    test_texts = test_data['text'].tolist()
    test_labels = test_data['label'].tolist()
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


if __name__ == '__main__':
    set_seed(22)
    configs = [
        {'model_name': 'bert', 'dataset': 'white', 'lr': 1e-5},
        {'model_name': 'distilbert', 'dataset': 'white', 'lr': 1e-5},
        {'model_name': 'roberta', 'dataset': 'white', 'lr': 1e-5},
        {'model_name': 'xlnet', 'dataset': 'white', 'lr': 1e-5},
        {'model_name': 'albert', 'dataset': 'white', 'lr': 1e-5},
    ]

    FREEZE_FUNCS = {
        'bert': freeze_bert_layers,
        'distilbert': freeze_distilbert_layers,
        'roberta': freeze_roberta_layers,
        'xlnet': freeze_xlnet_layers,
        'albert': freeze_albert_layers
    }

    for cfg in configs:
        print(f"\nTraining {cfg['model_name']} with lr={cfg['lr']}")

        file_dirs = {
            'bert': './model/bert-base-uncased',
            'distilbert': './model/distilbert-base-uncased',
            'roberta': './model/roberta-base',
            'xlnet': './model/xlnet-base-cased',
            'albert': './model/albert-base-v2'
        }
        file_dir = file_dirs[cfg["model_name"]]
        save_path = f'./STmodel/{cfg["dataset"]}/{cfg["model_name"]}'
        os.makedirs(save_path, exist_ok=True)

        model = AutoModelForSequenceClassification.from_pretrained(file_dir, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(file_dir)
        tokenizer.save_pretrained(save_path)

        freeze_fn = FREEZE_FUNCS.get(cfg["model_name"])
        freeze_fn(model)

        # data
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = get_data(cfg["dataset"])
        train_loader = DataLoader(MyDataset(train_texts, train_labels, tokenizer), batch_size=32, shuffle=True)
        val_loader = DataLoader(MyDataset(val_texts, val_labels, tokenizer), batch_size=64)
        test_loader = DataLoader(MyDataset(test_texts, test_labels, tokenizer), batch_size=64)

        # Compute class weights
        counts = Counter(train_labels)
        total = sum(counts.values())
        weights = [total / counts[i] for i in sorted(counts.keys())]
        class_weights = torch.tensor(weights, dtype=torch.float).cuda()

        # train
        train(model, train_loader, val_loader, lr=cfg["lr"], num_epochs=30, save_path=save_path,
              class_weights=class_weights)
        # Test the saved model
        model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=2)
        test(model, test_loader)
