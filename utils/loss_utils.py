from collections import Counter
import logging

import torch
import torch.nn as nn


def calc_class_weights(train_data):
    """
    Calculate inverse class weights for each class in the training data
    @param train_data: a batch generator (utils/batch_utils.py)
    @return: weights, a torch tensor of class weights
    """
    labels = [t[-1] for t in train_data]
    counts = Counter(labels)
    weights = torch.zeros(len(counts))

    for c in counts:
        weights[c] = counts[c]
    # return the weights, which are the normalized inverse class frequency (1/frequency(label))
    weights = weights.pow(-1)
    weights = weights / weights.sum()

    return weights


def get_loss_functions(datasets, is_multilabel, device, kwargs):
    """
    Calculate multiple loss functions, in case we want different class weights for each
    @param datasets: a list of datasets, actually not really useful here at all, come to think of it
    @param is_multilabel: a list of booleans specifying whether each dataset is multilabel or not
    @param device: the torch device our tensors should be on
    @param kwargs: kwargs generated by main.py
    @return: loss_function, a torch loss function
    """
    loss_functions = []

    # convert multilabel info into a list for ease of traversal
    if isinstance(is_multilabel, int):
        is_multilabel = [is_multilabel]

    for d, multi in zip(datasets, is_multilabel):
        # multilabel problems get sigmoid bce
        if multi:
            if kwargs.class_weights:
                logging.warning("Can't use class weights with multilabel output. Ignoring weights.")
            ls = nn.BCEWithLogitsLoss(reduction="mean").to(device)
        elif kwargs.class_weights:
            ls = nn.CrossEntropyLoss(weight=calc_class_weights(d),
                                     reduction="mean").to(device)
        else:
            ls = nn.CrossEntropyLoss(reduction="mean").to(device)

        loss_functions.append(ls)

    return loss_functions


def get_simultaneous_loss_functions(dataset, is_multilabel, device, kwargs):
    """
    calculate multiple loss functions, in case we want different class weights for each
    @param dataset: ignored, for compatibility
    @param is_multilabel: a boolean specifying whether this dataset is multilabel
    @param device: the torch device our tensors should be on
    @param kwargs: kwargs generated by main.py
    @return: loss_function, a torch loss function
    """
    loss_functions = []

    for multi in is_multilabel:
        # multilabel problems get sigmoid bce
        if multi:
            if kwargs.class_weights:
                logging.warning("Can't use class weights with multilabel output. Ignoring weights.")
            ls = nn.BCEWithLogitsLoss(reduction="mean").to(device)
        elif kwargs.class_weights:
            raise NotImplementedError("Don't know how to do class weights with multi-task datasets yet")
        else:
            ls = nn.CrossEntropyLoss(reduction="mean").to(device)

        loss_functions.append(ls)

    return loss_functions


class LossCalculator:
    def __init__(self, loss_fns):
        """
        A class which helps in calculating loss for complex tasks. This is the simplest version.
        @param loss_fns: a list of callables which will calculate loss given (preds, golds)
        """
        # loss function itself keeps the class weights I think
        self.loss_fns = loss_fns

    def get_loss(self, preds, golds, dataset_id=0):
        """
        Given predictions and gold labels, calculate the loss for one batch of one task according to our own dictionary.
        @param preds: a tensor of predicted labels
        @param golds: a tensor of gold labels, the same shape as preds
        @param dataset_id: the index of this dataset in the model (e.g., 0 for stress, 1 for emotion)
        @return: loss, a tensor of calculated loss (usually a scalar)
        """
        return self.loss_fns[dataset_id](preds, golds)


class SimultaneousLossCalculator(LossCalculator):
    def __init__(self, loss_fns, stress_weight=0.5):
        """
    A class which helps in calculating loss for the Multi model
        @param loss_fns: a list of callables which will calculate loss given (preds, golds)
        @param stress_weight: a float (0-1) that specifies how much weight the stress task should be given
        """
        super().__init__(loss_fns)

        if stress_weight == -1:
            self.lambdas = [1, 1]
        else:
            self.lambdas = [stress_weight, 1 - stress_weight]

    def get_loss(self, preds, golds, dataset_id=0):
        """
        Given predictions and gold labels, calculate the loss according to our own dictionary.
        @param preds: a tensor of predicted labels
        @param golds: a tensor of gold labels, the same shape as preds
        @param dataset_id: the index of this dataset in the model (e.g., 0 for stress, 1 for emotion)
        @return: loss, a tensor of calculated loss (usually a scalar)
        """
        return (self.lambdas[0] * self.loss_fns[0](preds[0], golds[0])) + \
               (self.lambdas[1] * self.loss_fns[1](preds[1], golds[1]))
