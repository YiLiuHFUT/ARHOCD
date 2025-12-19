import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.distributions import LogNormal
from tools import cal_metrics
import os
import random
from itertools import combinations, product


def normalize(array):
    """
    Min-max normalization of a numpy array.

    Parameters
    ----------
    array : np.ndarray
        Input array to normalize.

    Returns
    -------
    np.ndarray
        Normalized array in the range [0, 1].
    """
    epsilon = 1e-8
    array_min = np.min(array)
    array_max = np.max(array)
    list_normalized = (array - array_min) / (array_max - array_min + epsilon)
    return list_normalized


def calculate_weights(pred_prob, alpha, belta):
    """
    Parameters
    ----------
    pred_prob : np.ndarray
        Prediction probabilities with shape (num_samples, num_variants, num_models, num_classes)
    alpha : float
        Scaling factor for variant weights
    beta : float
        Scaling factor for model weights

    Returns
    -------
    np.ndarray
        Weight matrix with shape (num_samples, num_variants, num_models)
    """
    num_samples, num_variants, num_models, num_classes = pred_prob.shape
    weight_matrix = np.ones((num_samples, num_variants, num_models))

    for i in range(num_samples):
        # ---------- (1) Calculate variant weights w_v----------
        variant_avg_distances = np.zeros(num_variants)
        for v in range(num_variants):
            class1_probs = pred_prob[i, v, :, 1]  # Probabilities for class 1 across models
            variant_avg_distances[v] = np.var(class1_probs, ddof=1)

        variant_normalized_distances = normalize(variant_avg_distances)
        w_v = np.exp(-alpha * variant_normalized_distances)
        w_v = w_v / np.sum(w_v)

        # ---------- (2) Calculate model weights w_m----------
        model_avg_distances = np.zeros(num_models)
        for m in range(num_models):
            class1_probs = pred_prob[i, :, m, 1]
            model_avg_distances[m] = np.var(class1_probs, ddof=1)

        model_normalized_distances = normalize(model_avg_distances)
        w_m = np.exp(-belta * model_normalized_distances)
        w_m = w_m / np.sum(w_m)

        # ---------- 3. Combine variant and model weights ----------
        weight_matrix[i] = (w_v[:, None] + w_m[None, :]) / (num_variants + num_models)
    return weight_matrix


def compute_kl_div(posterior_mu, posterior_sigma, prior_mu, prior_sigma):
    """
    Parameters
    ----------
    posterior_mu : torch.Tensor
        Mean of posterior distribution [batch_size, num_samples, num_models]
    posterior_sigma : torch.Tensor
        Std of posterior distribution
    prior_mu : torch.Tensor
        Mean of prior distribution
    prior_sigma : torch.Tensor
        Std of prior distribution

    Returns
    -------
    torch.Tensor
        Mean KL divergence across the batch
    """
    eps = 1e-8
    posterior_sigma = torch.clamp(posterior_sigma, min=eps)
    prior_sigma = torch.clamp(prior_sigma, min=eps)

    kl_div = torch.log(prior_sigma / posterior_sigma) + (posterior_sigma ** 2 + (posterior_mu - prior_mu) ** 2) / (
            2 * prior_sigma ** 2) - 0.5

    return kl_div.sum(dim=(1, 2)).mean()


def reparameterize_log_normal(mu, sigma, num_samples=100):
    """
    Parameters
    ----------
    mu : torch.Tensor
        Log-space mean [batch_size, num_samples, num_models]
    sigma : torch.Tensor
        Log-space std
    num_samples : int
        Number of Monte Carlo samples

    Returns
    -------
    torch.Tensor
        Estimated expectation of log-normal samples [batch_size, num_samples, num_models]
    """
    torch.manual_seed(4)
    epsilon = torch.randn(mu.size(0), mu.size(1), mu.size(2), num_samples, device=mu.device)
    mu = mu.unsqueeze(-1)  # [B, S, M, 1]
    sigma = sigma.unsqueeze(-1)
    z = mu + sigma * epsilon
    samples = torch.exp(z)
    return samples.mean(dim=-1)


def train_aggre_bayesian(model, val_loader, test_loader, path, alpha, beta, dataset, num_samples=6, num_models=5,
                         lr=1e-4, lamda=0.0001, class_weights=None, sigma=1):
    """
    Parameters
    ----------
    model : nn.Module
        Aggregation model
    val_loader : DataLoader
        Validation dataset loader
    test_loader : DataLoader
        Test dataset loader
    path : str
        Path to save the trained model
    alpha, beta : float
        Scaling factors for variant/model weights
    dataset : str
        Dataset name
    num_samples : int
        Number of MC samples
    num_models : int
        Number of base models
    lr : float
        Learning rate
    lamda : float
        KL loss scaling factor
    class_weights : torch.Tensor
        Optional class weights for NLL loss
    sigma : float
        Standard deviation for prior/posterior log-normal distributions
    """
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    model.train().cuda()
    best_f, best_loss = 0.0, float('inf')
    for epoch in tqdm(range(60), desc="Training Epochs"):
        total_loss = 0.0
        for batch in val_loader:
            pro, label = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()

            # ---------- Prior weight estimation ----------
            prior_weight = torch.tensor(
                calculate_weights(pro.cpu().numpy(), alpha=alpha, belta=beta)).cuda()
            prior_sigma = torch.ones(pro.size(0), pro.size(1), pro.size(2)).cuda() * sigma
            prior_mu = (torch.log(prior_weight) - (prior_sigma ** 2) / 2).cuda()

            # ---------- Posterior estimation ----------
            posterior_weight = model(pro[..., 1])
            posterior_sigma = torch.ones(pro.size(0), pro.size(1), pro.size(2)).cuda() * sigma
            posterior_mu = (torch.log(posterior_weight) - (posterior_sigma ** 2) / 2).cuda()

            # ---------- KL divergence ----------
            kl_loss = compute_kl_div(posterior_mu, posterior_sigma, prior_mu, prior_sigma)

            # ---------- Monte Carlo aggregation ----------
            weight_matrix = reparameterize_log_normal(posterior_mu, posterior_sigma, num_samples=200)
            weight_sum = weight_matrix.view(weight_matrix.size(0), -1).sum(dim=1, keepdim=True)
            weight_matrix = (weight_matrix.view(weight_matrix.size(0), -1) /
                             weight_sum).view(weight_matrix.size(0), num_samples, num_models)
            final_output = torch.sum(pro * weight_matrix[:, :, :, None], dim=(1, 2))

            loss = criterion(torch.log(final_output), label) + lamda * kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch}, Avg Loss: {total_loss / len(val_loader):.5f}')
        scheduler.step()

        # ---------- Evaluate ----------
        test_loss, test_acc, test_p2, test_r2, test_f2 = test_aggre_bayesian(model, test_loader, MC=True)
        print('(Aggre) Test loss', test_loss)
        print('(Aggre) Test acc, p2, r2, f2', test_acc, test_p2, test_r2, test_f2)

        # ---------- Save best model ----------
        if best_f < test_f2 or (best_f == test_f2 and best_loss > test_loss):
            best_f = test_f2
            best_loss = test_loss
            print(f'Save model! Best test loss is {test_loss:.5f}')
            torch.save(model.state_dict(), os.path.join(path, f'{dataset}.pt'))


def test_aggre_bayesian(model, test_loader, MC=True, return_per_class_acc=False):
    """
    Parameters
    ----------
    model : nn.Module
    test_loader : DataLoader
    MC : bool
        If True, use Monte Carlo sampling; else use expectation
    return_per_class_acc : bool
        Whether to return per-class accuracy

    Returns
    -------
    tuple
        Loss, accuracy, precision, recall, F2, and optionally per-class accuracy
    """
    criterion = nn.NLLLoss()
    model.eval().cuda()

    predictions, all_labels, probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            pro, labels = batch[0].cuda(), batch[1].cuda()
            posterior_weight = model(pro[..., 1])
            posterior_sigma = torch.ones(pro.size(0), pro.size(1), pro.size(2)).cuda()
            posterior_mu = (torch.log(posterior_weight) - (posterior_sigma ** 2) / 2).cuda()
            posterior = LogNormal(posterior_mu, posterior_sigma)
            if MC:
                weight_matrix = posterior.sample((200,)).mean(dim=0)
            else:
                weight_matrix = posterior_weight

            final_output = torch.sum(pro * weight_matrix[:, :, :, None], dim=(1, 2))
            loss = criterion(torch.log(final_output), labels)
            total_loss += loss.item()

            probs.extend(final_output.detach().cpu().numpy())
            predictions.extend(np.argmax(final_output.detach().cpu().numpy(), axis=1))
            all_labels.extend(labels.cpu().numpy())

    acc, p2, r2, f2, auc, acc0, acc1 = cal_metrics(all_labels, predictions, probs, return_per_class_acc=True)

    if return_per_class_acc:
        return total_loss / len(test_loader), acc, p2, r2, f2, acc0, acc1
    else:
        return total_loss / len(test_loader), acc, p2, r2, f2
