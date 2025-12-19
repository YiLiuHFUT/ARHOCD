from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def cal_metrics(y_true, y_predict, y_score=None, return_per_class_acc=False):
    """
        Args:
        y_true (np.ndarray): True labels.
        y_predict (np.ndarray): Predicted labels.
        y_score (np.ndarray, optional): Predicted scores (for ROC AUC computation).
        return_per_class_acc (bool, optional): If True, returns per-class accuracy.
    """

    # Ensure the lengths match
    if len(y_true) != len(y_predict):
        raise ValueError("The length of y_true and y_predict must be the same.")

    acc = accuracy_score(y_true, y_predict)
    p2 = precision_score(y_true, y_predict, average='macro')
    r2 = recall_score(y_true, y_predict, average='macro')
    f2 = f1_score(y_true, y_predict, average='macro')

    # Calculate confusion matrix and per-class accuracy
    cm = confusion_matrix(y_true, y_predict)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    acc_0 = per_class_acc[0]
    acc_1 = per_class_acc[1]

    y = []
    for i in range(len(y_true)):
        if y_true[i] == 0 or y_true[i] == 0.0:
            y.append([1, 0])
        else:
            y.append([0, 1])
    if y_score is not None:
        auc = roc_auc_score(y, y_score)
    else:
        auc = None

    if return_per_class_acc:
        return acc, p2, r2, f2, auc, acc_0, acc_1
    else:
        return acc, p2, r2, f2, auc


def load_model(file_dir, model_max_length):
    model = AutoModelForSequenceClassification.from_pretrained(file_dir)
    tokenizer = AutoTokenizer.from_pretrained(file_dir, model_max_length=model_max_length)
    return model, tokenizer


def freeze_bert_layers(model, num_freeze=6):
    for name, param in model.named_parameters():
        if name.startswith("bert.encoder.layer."):
            layer_num = int(name.split('.')[3])
            param.requires_grad = layer_num >= num_freeze
        elif name.startswith("bert.pooler") or name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = True


def freeze_distilbert_layers(model, num_freeze=3):
    for name, param in model.named_parameters():
        if name.startswith("distilbert.transformer.layer."):
            layer_num = int(name.split('.')[3])
            if layer_num < num_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif name.startswith("pre_classifier") or name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = True


def freeze_roberta_layers(model, num_freeze=6):
    for name, param in model.named_parameters():
        if name.startswith("roberta.encoder.layer."):
            layer_num = int(name.split('.')[3])
            if layer_num < num_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = True


def freeze_xlnet_layers(model, num_freeze=6):
    for name, param in model.named_parameters():
        if name.startswith("transformer.layer."):
            layer_num = int(name.split('.')[2])
            if layer_num < num_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif name.startswith("sequence_summary") or name.startswith("logits_proj"):
            param.requires_grad = True
        else:
            param.requires_grad = True


def freeze_albert_layers(model):
    for name, param in model.named_parameters():
        if name.startswith("albert.embeddings."):
            param.requires_grad = True
        else:
            param.requires_grad = True
