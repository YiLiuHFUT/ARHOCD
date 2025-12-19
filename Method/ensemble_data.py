from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tools import cal_metrics
import os


class PredictionDataset(Dataset):
    def __init__(self, positive_probs, labels):
        """
        Each sample contains:
        - positive_probs: prediction probabilities from multiple models and variants
                          Shape: [num_variants, num_models] (e.g., 6x5)
        - label: ground-truth class label (scalar)
        """
        self.data = torch.FloatTensor(positive_probs)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def test_aggre(model, test_loader, return_per_class_acc=False):
    criterion = nn.NLLLoss()
    model = model.cuda()
    model.eval()

    predictions, all_labels, probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pro = batch[0].cuda()
            labels = batch[1].cuda()
            weights = model(pro[..., 1])
            final_output = torch.sum(pro * weights[:, :, :, None], dim=(1, 2))
            loss = criterion(torch.log(final_output), labels)
            total_loss += loss.item()

            probs.extend(final_output.detach().cpu().numpy())
            predictions.extend(np.argmax(final_output.detach().cpu().numpy(), axis=1))
            all_labels.extend(labels.cpu().numpy())
    acc, p2, r2, f2, auc, acc0, acc1 = cal_metrics(all_labels, predictions, probs, return_per_class_acc=True)
    # print(f'Test loss: {total_loss / len(test_loader)}')
    # print('Test acc, p2, r2, f2, auc', acc, p2, r2, f2, auc)
    if return_per_class_acc:
        return total_loss / len(test_loader), acc, p2, r2, f2, acc0, acc1
    else:
        return total_loss / len(test_loader), acc, p2, r2, f2


def train_aggre(model, val_loader, test_loader, path, dataset, lr=1e-4, class_weights=None):
    """
    Parameters
    ----------
    model : nn.Module
        Aggregation model to be trained.
    val_loader : DataLoader
        DataLoader for validation dataset.
    test_loader : DataLoader
        DataLoader for test dataset.
    path : str
        Directory path to save the best model.
    dataset : str
        Dataset name (used for saving model file).
    lr : float
        Learning rate for Adam optimizer.
    class_weights : torch.Tensor or None
        Optional class weights for NLLLoss.
    num_epochs : int
        Number of training epochs.
    """
    criterion = nn.NLLLoss(class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    best_f, best_loss = 0.0, float('inf')
    for epoch in range(60):
        model.train().cuda()
        total_loss = 0.0

        for batch in tqdm(val_loader, desc=f"Training Epoch {epoch}"):
            pro, labels = batch[0].cuda(), batch[1].cuda()

            optimizer.zero_grad()
            weights = model(pro[..., 1])
            final_output = torch.sum(pro * weights[:, :, :, None], dim=(1, 2))

            loss = criterion(torch.log(final_output), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'=============epoch: {epoch}, loss: {total_loss / len(val_loader)}=============')
        scheduler.step()

        # Evaluate on test set
        test_loss, test_acc, test_p2, test_r2, test_f2 = test_aggre(model, test_loader)

        # Save best model based on F1 score
        if best_f < test_f2 or (best_f == test_f2 and best_loss > test_loss):
            best_f = test_f2
            best_loss = test_loss
            print(f'Save model! Best test loss is {test_loss:.5f}')
            torch.save(model.state_dict(), os.path.join(path, f'{dataset}.pt'))
