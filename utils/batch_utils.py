import abc
import random

import numpy as np

import torch
import torch.nn.utils.rnn as rnnutils

from configs import const


def get_batch(dataset, batch_size, batch_index, is_multilabel, num_labels=2):
    """
    Pull a batch of a given size and location from a given dataset
    @param dataset: the dataset in question, a list of tuples from our data processing
    @param batch_size: int, the size of one batch
    @param batch_index: int, the index of the bath to retrieve (batch #0, 1, 2, ...)
    @param is_multilabel: boolean, whether this dataset is multilabel or not
    @param num_labels: int, the number of labels in this dataset
    @return: batch_tokens, batch_type_ids, batch_attn_mask, batch_labels
    """

    batch = dataset[batch_index * batch_size: min((batch_index + 1) * batch_size, len(dataset))]

    # pad inputs with padding value from const
    batch_tokens = rnnutils.pad_sequence([torch.tensor(b[0]) for b in batch],
                                         padding_value=const.PADDING_IDX,
                                         batch_first=True)

    # pad token types with 0
    batch_token_types = rnnutils.pad_sequence([torch.tensor(b[1]) for b in batch],
                                              padding_value=0, batch_first=True)

    # pad attention mask with 0 (this being the whole point of an attention mask)
    batch_attn_mask = rnnutils.pad_sequence([torch.tensor(b[2]) for b in batch],
                                            padding_value=0, batch_first=True)

    # label tensor is a list of class indices if single-label
    # and a "one-hot" (...k-hot?) encoding if multi-label
    if is_multilabel:
        batch_y = torch.zeros(len(batch), num_labels)

        # set the elements of the label tensor to 1 appropriately
        # this will totally ignore anything after b[3], which is fine for here
        for i, b in enumerate(batch):
            if isinstance(b[3], int):
                batch_y[i][b[3]] = 1
            else:
                for label in b[3]:
                    batch_y[i][label] = 1
    else:
        batch_y = torch.tensor([b[3][0] for b in batch])

    return batch_tokens, batch_token_types, batch_attn_mask, batch_y


def get_batch_multitask(dataset, batch_size, batch_index, is_multilabel, num_labels):
    """
    Pull a batch of a given size and location from a given dataset, for the Multi case
    @param dataset: the dataset in question, a list of tuples from our data processing
    @param batch_size: int, the size of one batch
    @param batch_index: int, the index of the bath to retrieve (batch #0, 1, 2, ...)
    @param is_multilabel: a list of booleans, whether each task is multilabel or not
    @param num_labels: a list of ints, the number of labels for each task in this dataset
    @return: batch_tokens, batch_type_ids, batch_attn_mask, batch_labels (a list of tensors now)
    """

    batch = dataset[batch_index * batch_size: min((batch_index + 1) * batch_size, len(dataset))]

    # pad inputs with padding value from const
    batch_tokens = rnnutils.pad_sequence([torch.tensor(b[0]) for b in batch],
                                         padding_value=const.PADDING_IDX,
                                         batch_first=True)

    # pad token types with 0
    batch_token_types = rnnutils.pad_sequence([torch.tensor(b[1]) for b in batch],
                                              padding_value=0, batch_first=True)

    # pad attention mask with 0 (this being the whole point of an attention mask)
    batch_attn_mask = rnnutils.pad_sequence([torch.tensor(b[2]) for b in batch],
                                            padding_value=0, batch_first=True)

    # label tensor is a list of class indices if single-label
    # and a "one-hot" (...k-hot?) encoding if multi-label

    labels = []
    for j, im in enumerate(is_multilabel):
        if im:
            batch_y = torch.zeros(len(batch), num_labels[j])

            # set the elements of the label tensor to 1 appropriately
            # this will totally ignore anything after b[3], which is fine for here
            for i, b in enumerate(batch):
                if isinstance(b[3+j], int):
                    batch_y[i][b[3+j]] = 1
                else:
                    for label in b[3+j]:
                        batch_y[i][label] = 1
        else:
            batch_y = torch.tensor([b[3+j][0] for b in batch])

        labels.append(batch_y)

    return batch_tokens, batch_token_types, batch_attn_mask, labels


class BatchGenerator:
    """
    A class that takes in one or more datasets and creates batches.
    The batches may be shuffled or incorporate one or more datasets.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def init_epoch(self):
        """
        Resets any internal metrics for the next epoch, or calculates new metrics (e.g., proportions) for a new epoch
        """
        pass

    @abc.abstractmethod
    def get_batches(self):
        """
        A generator method that returns one or more batches at a time.
        """
        pass


class SimpleBatchGenerator(BatchGenerator):
    """
    A batch generator intended to work with a single dataset.
    """
    def __init__(self, datasets, batch_size, device, is_multilabel, num_labels, shuffle=True):
        """
        Create the batch generator
        @param datasets: a list of lists of data points (so should be one dataset wrapped in an extra list)
        @param batch_size: int, the desired size of one batch
        @param device: the torch device to use
        @param is_multilabel: boolean, whether this dataset is multilabel
        @param num_labels: int, the number of labels this dataset has
        @param shuffle: boolean, whether to shuffle the data (not required for dev/eval)
        """
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        # this should be a single Boolean
        self.is_multilabel = is_multilabel
        self.num_labels = num_labels

    def init_epoch(self):
        """
        No-op; this generator has nothing to reset or calculate.
        """
        pass

    def __len__(self):
        """
        Length function
        @return: an int, the length of this batch generator
        """
        return int(np.ceil(len(self.datasets[0]) / self.batch_size))

    def get_batches(self):
        """
        Generator that returns one batch at a time for one epoch. Shuffles data.
        """

        # sort the dataset from longest to shortest
        dataset = sorted(self.datasets[0], key=lambda x: len(x[0]), reverse=True)

        batch_idxes = np.arange(0, len(self))
        if self.shuffle:
            # shuffle the batches, but be sure to put the longest batch first
            # (shuffle batches so that batches still generally have all similar-length inputs)
            batch_idxes = np.insert(np.random.permutation(batch_idxes[1:]), 0, 0)

        for i in batch_idxes:
            # yield one batch at a time, where a batch is...
            # (token_ids, token_type_ids, attention_mask, golds)
            # golds is a list of [task_1_labels, task_2_labels, ...]
            batch_tokens, batch_token_types, batch_attn_mask, batch_y = get_batch(dataset, self.batch_size, i,
                                                                                  self.is_multilabel, self.num_labels)

            yield 0, batch_tokens.to(self.device), batch_token_types.to(self.device), \
                  batch_attn_mask.to(self.device), batch_y.to(self.device)


class RoundRobinBatchGenerator(BatchGenerator):
    """
    A batch generator for multiple datasets, which alternates between batches at every step.
    """
    def __init__(self, datasets, batch_size, device, is_multilabel, num_labels, shuffle=True):
        """
        Create the batch generator
        @param datasets: a list of lists of data points
        @param batch_size: int, the desired size of one batch
        @param device: the torch device to use
        @param is_multilabel: list of booleans, whether each dataset is multilabel
        @param num_labels: lost of ints, the number of labels each dataset has
        @param shuffle: boolean, whether to shuffle the data (not required for dev/eval)
        """
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        # this should be a list of Booleans
        self.is_multilabel = is_multilabel
        self.num_labels = num_labels

    def init_epoch(self):
        """
        No-op; this generator has nothing to reset or calculate.
        """
        pass

    def __len__(self):
        """
        Length function
        @return: an int, the length of this batch generator
        """
        return int(np.ceil(min([len(d) for d in self.datasets]) / self.batch_size) * len(self.datasets))

    def get_batches(self):
        """
        Generator that returns one batch at a time for one epoch. Shuffles data. Subsamples longer datasets.
        """

        # subsample longer datasets to match the shortest dataset exactly
        shortest_data_len = min([len(d) for d in self.datasets])
        datasets = [random.sample(dataset, shortest_data_len) for dataset in self.datasets]

        # sort them all from longest to shortest
        datasets = [sorted(dataset, key=lambda x: len(x[0]), reverse=True) for dataset in datasets]

        # create a unique batch idx permutation for each dataset if shuffling
        batch_idxes = [np.arange(0, len(self) / len(datasets)) for _ in datasets]

        if self.shuffle:
            # shuffle the batches, but be sure to put the longest batch first
            # (shuffle batches so that batches still generally have all similar-length inputs)
            batch_idxes = [np.insert(np.random.permutation(bi[1:]), 0, 0) for bi in batch_idxes]

        # for each timestep...
        for i in range(len(batch_idxes[0])):
            # for each dataset...
            for j, d in enumerate(datasets):
                # yield one batch from one dataset at a time
                batch_tokens, batch_token_types, batch_attn_mask, batch_y = get_batch(d, self.batch_size,
                                                                                      int(batch_idxes[j][i]),
                                                                                      self.is_multilabel[j],
                                                                                      self.num_labels[j])
                yield j, batch_tokens.to(self.device), batch_token_types.to(self.device), \
                      batch_attn_mask.to(self.device), batch_y.to(self.device)


class SimultaneousBatchGenerator(BatchGenerator):
    """
    A batch generator for one dataset with multiple tasks.
    """

    def __init__(self, dataset, batch_size, device, is_multilabel, num_labels, shuffle=True):
        """
        Create the batch generator
        @param dataset: a list of data points
        @param batch_size: int, the desired size of one batch
        @param device: the torch device to use
        @param is_multilabel: list of booleans, whether each task is multilabel
        @param num_labels: lost of ints, the number of labels each task has
        @param shuffle: boolean, whether to shuffle the data (not required for dev/eval)
        """
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        # this should be a list of Booleans
        self.is_multilabel = is_multilabel
        self.num_labels = num_labels

    def init_epoch(self):
        """
        No-op; this generator has nothing to reset or calculate.
        """
        pass

    def __len__(self):
        """
        Length function
        @return: an int, the length of this batch generator
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def get_batches(self):
        """
        Generator that returns one batch at a time for one epoch. Shuffles data.
        """

        # sort the data from longest to shortest
        dataset = sorted(self.dataset, key=lambda x: len(x[0]), reverse=True)

        # create a batch idx permutation for each dataset if shuffling
        batch_idxes = np.arange(0, len(self))

        if self.shuffle:
            # shuffle the batches, but be sure to put the longest batch first
            # (shuffle batches, not inputs, so that batches still generally have all similar-length inputs)
            batch_idxes = np.insert(np.random.permutation(batch_idxes[1:]), 0, 0)

        # for each timestep...
        for i in range(len(batch_idxes)):
            # yield one batch from the dataset at a time
            batch_tokens, batch_token_types, batch_attn_mask, batch_y = get_batch_multitask(dataset, self.batch_size,
                                                                                            int(batch_idxes[i]),
                                                                                            self.is_multilabel,
                                                                                            self.num_labels)
            yield None, batch_tokens.to(self.device), batch_token_types.to(self.device), \
                  batch_attn_mask.to(self.device), [y.to(self.device) for y in batch_y]
