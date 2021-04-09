import csv
import logging

from transformers import BertTokenizer


def sniff_multilabel(datasets):
    """
    Decide if one or more datasets are multilabel (i.e., they have at least one item with 2+ labels)
    @param datasets: datasets generated by get_datasets, i.e., a list of lists of tuples
    @return: is_multilabel, a list of lists of Booleans
    """

    is_multilabel = []

    for dataset in datasets:
        dataset_is_multilabel = []
        for task_idx in range(3, len(dataset[0])):
            task_is_multilabel = any([len(d[task_idx]) > 1 for d in dataset])
            dataset_is_multilabel.append(task_is_multilabel)

        is_multilabel.append(dataset_is_multilabel)

    return is_multilabel


def load_from_file(fn):
    """
    Load a single .csv (assumed to have only two columns: text, label)
    @param fn: string, the filename of the data file
    @return: data, list of datapoints (themselves lists)
    """

    # assumes header, therefore ignores first line
    # assumes text is in first column and any further columns are string labels, with multilabel delimited by ","
    with open(fn, "r", encoding="utf-8") as f:
        data = list(csv.reader(f))[1:]  # for debug, do [1:1000]

    return data


def get_label2idx(labels):
    """
    Create a label-ID correspondence.
    @param labels: a list of string labels
    @return: a dictionary of {"label": id} correspondences
    """

    label2idx = {}

    # labels should be basically a column of the spreadsheet as a list
    for label in labels:
        # handle multilabel case
        if "," in label:
            labels = [l.lower().strip() for l in label.split(",")]

            for lab in labels:
                if lab not in label2idx and len(lab) > 0:
                    label2idx[lab] = len(label2idx)
        else:
            label = label.lower()

            if label not in label2idx and len(label) > 0:
                label2idx[label] = len(label2idx)

    return label2idx


def numerify_dataset(dataset, tokenizer, label2idx, lower=True):
    """
    Turn a text dataset entirely into numbers
    @param dataset: the dataset returned from load_from_file, a list of lists (of text fields)
    @param tokenizer: a BertTokenizer or other equivalent callable
    @param label2idx: a dictionary of {"text_label": id} pairs from get_label2idx
    @param lower: ignored
    @return: processed_dataset, a list of tuples (which contain only numbers)
    """

    processed_dataset = []

    if len(range(1, len(dataset[0]))) > 1:
        logging.warning("you have passed in a dataset with multiple tasks by itself. you will encounter errors if you "
                        "use any model except \"multi\".")

    for datapoint in dataset:
        # truncation of long sequences takes place here
        processed_text = tokenizer(datapoint[0], truncation=True, max_length=tokenizer.max_len)

        processed_labels = []

        # process 1+ tasks for each datapoint
        for field_idx in range(1, len(datapoint)):
            processed_label = []

            # create a list of labels for this datapoint and task combo
            # (if not multilabel, spit out a singleton list such as [1])
            for label in datapoint[field_idx].split(","):
                if len(label) > 0:
                    processed_label.append(label2idx[field_idx-1][label.lower()])

            processed_labels.append(processed_label)

        processed_dataset.append(tuple([processed_text["input_ids"], processed_text["token_type_ids"],
                                        processed_text["attention_mask"]] + processed_labels))

    return processed_dataset


def get_datasets(filenames, tokenizer=None, label2idx=None, lower=True):
    """
    Can load one or more datasets from file and return them as a list
    @param filenames: a list of filenames to load
    @param tokenizer: a BertTokenizer or other equivalent callable that tokenizes text data
    @param label2idx: A list of dictionaries of {"label": id} correspondences (one for each filename)
    @param lower: True if the data should be lowercased. NOTE: a provided tokenizer overrides this setting.
    @return: data (a list of lists of tuples), tokenizer (the same as passed in; a default tokeniozer if None),
        label2idx (the same as passed in; computed from the data if None)
    """

    # load data from file into lists of ("text", "label1", "label2", ...)
    data = [load_from_file(f) for f in filenames]

    # the number of tasks in each dataset is the number of columns (-1 for the text)
    num_tasks = [len(d[0]) - 1 for d in data]

    # create tokenizer and label2idxes if not given
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=lower, do_basic_tokenize=True)

    # create a separate label2idx for each dataset and each of their tasks
    if label2idx is None or len(label2idx) < len(data):
        label2idx = [[get_label2idx([d[j] for d in dataset]) for j in range(1, num_tasks[i]+1)]
                     for i, dataset in enumerate(data)]

    # now process all the data. each datapoint is (token_ids, seq_ids, attn_mask, label)
    data = [numerify_dataset(d, tokenizer, l, lower) for d, l in zip(data, label2idx)]

    # data is now a list of lists of tuples, each tuple indicating a numerical datapoint
    return data, tokenizer, label2idx
