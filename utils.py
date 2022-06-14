import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pickle
import config




def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def save_checkpoint(data_name, epoch, epochs_since_improvement,decoder,decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param decoder: decoder model
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'decoder': decoder.module.state_dict(),
             'decoder_optimizer': decoder_optimizer}
    # filename = config.checkpoint_path + 'best_checkpoint_85_gt_w_1_loss_no_det.pth.tar'
    filename = config.checkpoint_path + 'best_checkpoint.pth.tar'

    # filename = config.checkpoint_path + 'best_checkpoint_normal.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def JointEmbeddingLoss(feature_emb1, feature_emb2):
    """
    embedding loss
    :param feature_emb1: prediction
    :param feature_emb2: gt
    :return:
    """
    batch_size = feature_emb1.size()[0]
    return torch.sum(
        torch.clamp(
            torch.mm(feature_emb1, feature_emb2.t()) - torch.sum(feature_emb1 * feature_emb2, dim=-1) + 1,
            min=0.0
        )
    ) / (batch_size * batch_size)


def one_hot(t, c):
    """
    :param t: phrase
    :param c: vocab_size
    :return:
    """
    return torch.zeros(*t.size(), c, device=t.device).scatter_(-1, t.unsqueeze(-1), 1)
