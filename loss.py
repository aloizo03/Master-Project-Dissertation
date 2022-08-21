'''This loss Class is taken by the GROUP_DRO loss class
link: https://github.com/kohpangwei/group_DRO/blob/master/loss.py and modified for the needs of this study'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random


class LossComputer:
    """
    This loss function is for the Group-DRO fairness  in-processing algorithm
    """
    def __init__(self, criterion, is_robust, dataset, device, alpha=None, gamma=0.1, adj=None, min_var_weight=0,
                 step_size=0.01, normalize_loss=False, btl=False):
        """
        This function is the initialization of the loss function class
        :param criterion: nn.Criterion, the criterion of the calculation of the loss and based this loss build the group-DRO(cross entropy)
        :param is_robust: bool, said if used the robust in the loss
        :param dataset: Dataset, determine the training dataset
        :param device: torch.device, said in which device run in gpu with cuda or cpu
        :param alpha: float, determine the value of the blt robust loss
        :param gamma: float, the value of the calculation of the previous weights for each sensitive feature
        :param adj: list, the adjustment from the adjustment generalization
        :param min_var_weight: float, the minimum value that can have the weight
        :param step_size: float, the value of the increasing the weights for the model and for the sensitive feature weights
        :param normalize_loss: bool, said if the loss must normalize
        :param btl: bool, said if add the btl loss counting in the loss function
        """
        self.device = device
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self._adj_ = adj

        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().to(self.device)
        self.group_frac = self.group_counts / self.group_counts.sum()

        self.processed_data_counts = torch.zeros(self.n_groups).to(self.device)
        # convert the adjustment of adjustment generalization into a tensor
        if self._adj_ is not None:
            self.adj = torch.from_numpy(self._adj_).float().to(self.device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(self.device)

        # with robust the alpha must have a value
        if is_robust:
            assert alpha, 'alpha must be specified'

            # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(self.device) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).to(self.device)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(self.device)

        self.update_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_batch_counts = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_loss = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_acc = torch.zeros(self.n_groups).to(self.device)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_dataset(self, dataset):
        """
        When a new task of classes added into the network update the dataset
        :param dataset: Dataset, the dataset class
        :return: None
        """
        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().to(self.device)
        self.group_frac = self.group_counts / self.group_counts.sum()

        self.processed_data_counts = torch.zeros(self.n_groups).to(self.device)
        if self._adj_ is not None:
            self.adj = torch.from_numpy(self._adj_).float().to(self.device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(self.device)

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(self.device) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).to(self.device)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(self.device)

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None):
        """
        Compute the loss and return it. The sensitive feature are counted if is selected
        the robust or if selected both robust and blt
        :param yhat: torch.Tensor, a tensor with all the predictions
        :param y: torch.Tensor, a tensor with all the ground truth labels
        :param group_idx: list, the sensitive feature group of each prediction
        :return: torch.Variable: the loss value
        """
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        """
        This function compute the robust loss based the group loss
        :param group_loss: torch.Tensor, the group loss for each predicted label take count the sensitive features
        :param group_count: torch.Tensor, The group count of the loss
        :return: torch.Tensor, Torch.tensor : The compute robust loss and the weights for each demographic group
        """
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        """
        This function compute the robust loss with the blt counting
        :param group_loss: torch.Tensor, the group loss for each predicted label take count the sensitive features
        :param group_count: torch.Tensor, The group count of the loss
        :return: torch.Tensor, Torch.tensor : The compute robust loss and the weights for each demographic group
        """
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        """
        This function compute the robust loss with a
        :param group_loss: torch.Tensor, the group loss for each predicted label take count the sensitive features
        :param ref_loss: torch.Tensor, the loss take count the adjustment generalization and the group count
                                       from the dataset for the sensitive features
        :return: torch.Tensor, Torch.tensor: The compute robust loss and the weights for each demographic group
        """
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        """
        This function compute observed counts and mean loss for each group
        :param losses: torch.Tensor, the losses from the Criterion function for each predicted label
        :param group_idx: list, the id of the demographic group for each predict label
        :return: torch.Tensor, torch.Tensor: the group loss and the group count
        """
        group_map = (torch.LongTensor(group_idx).to(self.device) == torch.arange(self.n_groups).unsqueeze(1).long().to(
            self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        """
        This function compute the exponential average loss for each group
        :param group_loss: torch.Tensor, the group loss for each predicted label take count the sensitive features
        :param group_count: torch.Tensor, The group count of the loss
        :return: None
        """
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)
        del prev_weights
        del curr_weights

    def reset_stats(self):
        """
        This function reset the stat for the loss
        :return: None
        """
        self.processed_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_batch_counts = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_loss = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_acc = torch.zeros(self.n_groups).to(self.device)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        """
        This function update the stat for each bach size step
        :param actual_loss: torch.Variable, the actual loss after the calculations
        :param group_loss: torch.Tensor, the group loss where take count the loss and the demographic group attribute
        :param group_acc: torch.tensor, the group accuracy from the prediction
        :param group_count: torch.Tensor, the group count
        :param weights: torch.Tensor, the weight for each demographic group
        :return: None
        """
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss

        # counts
        self.processed_data_counts += group_count
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, stats_dict, weight_decay=0.0005):
        """
        Return the model updated status into the dictionary
        :param model: nn.Module, the model class
        :param stats_dict: dict: the state dictionary of the model
        :param weight_decay: float, the weight decay of the SGD
        :return: dict: The update model status dictionary
        """
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = weight_decay / 2 * model_norm_sq.item()
        return stats_dict
