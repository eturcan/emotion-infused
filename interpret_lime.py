# native python
from itertools import chain

# science, math libraries
from scipy.stats import pearsonr
from tqdm import tqdm

# transformers and pytorch
import torch.nn.utils.rnn as rnnutils
from transformers import BertTokenizer

# LIME
from lime.lime_text import LimeTextExplainer

# our files
from main import *
from models.baselines import BertSingletask
from models.multitask import BertMultitask
from models.simultaneous_multitask import SimultaneousMultitask


secondary_tasks = {
    "goemotions-a": "data/goemotions/dev.csv",
    "goemotions-e": "data/goemotions/ekman_dev.csv",
    "goemotions-s": "data/goemotions/sentiment_dev.csv",
    "goemotions-v": "data/goemotions/vent_dev.csv",
    "vent": "data/vent/vent_dev.csv"
}


# used for LIME: take in text, output a probability distribution
def predict_single(text, model, tokenizer):
    """
    Predict the output of a single text string given a particular model
    @param text: the string to use as input
    @param model: the model to use for predicting
    @param tokenizer: the tokenizer to process the data
    @return: preds, the probability distribution over labels for this example with this model
    """
    # encode the text using our tokenizer (this is now a tensor of indices)
    processed_text = tokenizer(text, truncation=True, max_length=tokenizer.max_len)

    token_ids = processed_text["input_ids"]
    token_type_ids = processed_text["token_type_ids"]
    attn_mask = processed_text["attention_mask"]

    # pad the tensors if we have more than one input
    if isinstance(text, list) and len(text) > 1:
        # pad tokens
        token_ids = rnnutils.pad_sequence([torch.tensor(t, requires_grad=False) for t in token_ids],
                                          padding_value=const.PADDING_IDX, batch_first=True)
        # pad token types with 0
        token_type_ids = rnnutils.pad_sequence([torch.tensor(t, requires_grad=False) for t in token_type_ids],
                                               padding_value=0, batch_first=True)

        # pad attention mask with 0 (this being the whole point of an attention mask)
        attn_mask = rnnutils.pad_sequence([torch.tensor(a, requires_grad=False) for a in attn_mask], padding_value=0,
                                          batch_first=True)
    else:
        # otherwise, just turn them into tensors
        token_ids = torch.tensor(token_ids, requires_grad=False).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids, requires_grad=False).unsqueeze(0)
        attn_mask = torch.tensor(attn_mask, requires_grad=False).unsqueeze(0)

    # multi-alt and multi models need a dataset ID as well
    if not isinstance(model, BertSingletask):
        preds = model(0, token_ids, token_type_ids, attn_mask)
    else:
        preds = model(token_ids, token_type_ids, attn_mask)

    if len(preds) > 1 and isinstance(preds, list):
        preds = preds[0]

    # usually crossentropy does this for us, but convert into a probability distribution
    return preds.softmax(dim=1).detach()


def do_lime(model, model_path, data, tokenizer, idx2label):
    """
    Run LIME on an entire dataset and collect its large-magnitude and frequent rationales
    @param model: the model to use for predicting
    @param model_path: the path to which the model was saved, for naming purposes
    @param data: the data to use LIME on (dictionary form)
    @param tokenizer: the tokenizer to process the data
    @param idx2label: the label-ID correspondence for this dataset and model
    """
    # run LIME
    def lime_predict(text):
        return predict_single(text, model, tokenizer)

    explainer = LimeTextExplainer(class_names=["notstress", "stress"])

    # collect all rationales from the test set (50 for each data point)
    all_rationales = defaultdict(list)
    per_datapoint_rationales = []

    # rationalize the whole test set
    logging.info("explaining...")
    for datapoint in tqdm(data):
        explanation = explainer.explain_instance(datapoint["text"], lime_predict, num_samples=64,
                                                 num_features=50).as_list()

        for word, score in explanation:
            all_rationales[word.lower()].append(score)

        # also save the rationales for each data point
        pred = idx2label[predict_single(datapoint["text"], model, tokenizer).argmax().item()]
        per_datapoint_rationales.append({"input": datapoint["text"],
                                         "prediction": pred,
                                         "gold": datapoint["emotion" if "emotion" in datapoint.keys() else "emotions"],
                                         "explanation": " ".join([exp for exp, _ in explanation[:10]])})

    logging.info("done!")

    # print most common and most important rationales
    rationales = []
    for key in all_rationales:
        rationales.append((key, (sum(all_rationales[key]) / len(all_rationales[key])), len(all_rationales[key])))

    most_frequent = sorted(rationales, key=lambda x: x[2], reverse=True)[:50]
    largest_magnitude = sorted(rationales, key=lambda x: abs(x[1]), reverse=True)[:50]

    with open("analysis/{model}-frequent-rationales.txt".format(model=model_path[model_path.find("/") + 1:]), "w+",
              encoding="utf-8") as f:
        for r in most_frequent:
            f.write(r[0] + "\n")

    with open("analysis/{model}-large-rationales.txt".format(model=model_path[model_path.find("/") + 1:]), "w+",
              encoding="utf-8") as f:
        for r in largest_magnitude:
            f.write(r[0] + "\n")

    # output rationales per data point to a file
    with open("analysis/{model}-lime.csv".format(model=model_path[model_path.find("/") + 1:]), "w+", encoding="utf-8",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "prediction", "gold", "explanation"])
        writer.writeheader()
        for item in per_datapoint_rationales:
            writer.writerow(item)


class SwappableBatchGenerator(butils.SimpleBatchGenerator):
    def __init__(self, datasets, batch_size, device, is_multilabel, num_labels, preferred_task_index, shuffle=False):
        """
        Create a BatchGenerator that can change which task ID it uses for any given dataset
        @param datasets: a single dataset (i.e., a big list)
        @param batch_size: an int
        @param device: a torch.device
        @param is_multilabel: one single boolean
        @param num_labels: a single int
        @param preferred_task_index: an int
        @param shuffle: a boolean
        """
        super(SwappableBatchGenerator, self).__init__(datasets, batch_size, device, is_multilabel, num_labels, shuffle)

        self.preferred_task_index = preferred_task_index

    def get_batches(self):
        """
        Returns one batch at a time for one epoch. Shuffles data.
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
            batch_tokens, batch_token_types, batch_attn_mask, batch_y = butils.get_batch(dataset, self.batch_size, i,
                                                                                         self.is_multilabel,
                                                                                         self.num_labels)

            yield self.preferred_task_index, batch_tokens.to(self.device), batch_token_types.to(self.device), \
                  batch_attn_mask.to(self.device), batch_y.to(self.device)


def predict_with_gold(model, data_loader, multi_flag=False):
    """
    Evaluate the model on the given data and return its scores
    @param model: the model to be evaluated
    @param data_loader: a BatchGenerator for the dataset to be evaluated
    @return: all_golds (predicted stress and emotion labels), all_preds (a list of gold labels)
    """

    model.eval()
    all_golds = []
    all_preds = []

    with torch.no_grad():
        # run through all batches in train generator
        for j, (dataset_id, tokens, token_types, attn_mask, golds) in enumerate(data_loader.get_batches()):
            # get model predictions
            if not multi_flag:
                preds = model.predict(dataset_id, tokens, token_types, attn_mask)
                all_golds.extend(golds.tolist())
                all_preds.extend(preds.tolist())
            else:
                stress, emo = model(tokens, token_types, attn_mask)

                stress = stress.argmax(dim=1)

                if model.is_multilabel[1]:
                    emo = (emo.float().sigmoid() > const.MULTILABEL_THRESHOLD) * 1
                else:
                    emo = emo.argmax(dim=1)

                all_golds.extend(golds[0].tolist())
                all_preds.append([stress, emo])

    return all_golds, all_preds


def correlation_ratio(stress_labels, emotion_labels):
    """
    Calculate the correlation ration eta between two sets of labels
    @param stress_labels: torch.tensor
    @param emotion_labels: torch.tensor
    @return eta (correlation ratio, scalar), y_avg_array (breakdown by class)
    """

    # implementation used: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    # this whole thing will likely throw an error if some emotion category was never present
    num_emotion_categories = np.max(emotion_labels) + 1
    y_avg_array = np.zeros(num_emotion_categories)
    n_array = np.zeros(num_emotion_categories)

    for i in range(0, num_emotion_categories):
        stress_for_this_emotion = stress_labels[np.argwhere(emotion_labels==i).flatten()]
        n_array[i] = len(stress_for_this_emotion)
        y_avg_array[i] = np.average(stress_for_this_emotion)

    y_total_avg = (np.multiply(y_avg_array, n_array).sum() / n_array.sum()).sum()
    numerator = np.multiply(n_array, np.power(y_avg_array - y_total_avg, 2)).sum()
    denominator = np.power(emotion_labels - y_total_avg, 2).sum()

    if numerator == 0:
        eta = 0.
    else:
        # this returns eta, not eta^2
        eta = np.sqrt(numerator / denominator)

    return eta, y_avg_array


def multiple_correlation_coefficient(stress_labels, emotion_labels):
    """
    Calculate the multiple correlation coefficient R^2
    @param stress_labels: torch.tensor
    @param emotion_labels: torch.tensor
    @return: R2 (scalar), c (breakdown by class)
    """
    c = np.array([pearsonr(stress_labels, emotion_labels[:, i])[0] for i in range(emotion_labels.shape[1])])
    nonnan_idxes = np.argwhere(~np.isnan(c)).flatten()

    # create Rxx, the correlation matrix of the emotion labels among themselves
    # Rxx should be symmetric under pearsonr
    Rxx = np.array([[pearsonr(emotion_labels[:, i], emotion_labels[:, j])[0] for i in range(emotion_labels.shape[1])]
                    for j in range(emotion_labels.shape[1])])

    # squash these matrices down to remove any nans from constant labels
    # constant labels give us no information so don't even look at them
    if len(nonnan_idxes) < len(c):
        print(nonnan_idxes)
        c = c[nonnan_idxes]
        Rxx = Rxx[nonnan_idxes[:, None], nonnan_idxes]

    try:
        # create R2, the correlation of stress and emotion
        R2 = np.dot(np.dot(c, np.linalg.inv(Rxx)), c)
    except np.linalg.LinAlgError:
        R2 = np.dot(c, c)

    return R2, c


def save_predictions(data_fn, preds, idx2label, model_name):
    """
    Save predictions to file
    @param data_fn: the file to predict on
    @param preds: the predictions to write
    @param idx2label: a label-ID correspondence dictionary
    @param model_name: the name of the model to use (for filename purposes)
    """
    # save the labels in text form
    labels = []
    for pred in preds:
        if isinstance(pred, list):
            labels.append(",".join([idx2label[i] for i, val in enumerate(pred) if val == 1]))
        else:
            labels.append(idx2label[pred])

    with open(data_fn, "r", encoding="utf-8") as f:
        input_data = list(csv.reader(f))[1:]

    with open("analysis/{m}-dreaddit-emo-preds.csv".format(m=model_name[model_name.find("/")+1:]), "w+",
              encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emotions"])

        for datapoint, this_labels in zip(input_data, labels):
            writer.writerow([datapoint[0], this_labels])


def do_task_swapping(model, model_path, primary_data_path, tokenizer, primary_label2idx, batch_size=16):
    """
    Perform multitask task-swapping analysis
    @param model: the model to use
    @param model_path: the model's filepath (for naming purposes)
    @param primary_data_path: the filename of the primary task data (stress)
    @param tokenizer: a huggingface tokenizer or equivalent callable
    @param primary_label2idx: the label2idx dictionary for the primary task
    @param batch_size: the batch size, int
    """
    # select the proper other task
    other_task = None
    for key in secondary_tasks:
        if key in model_path:
            other_task = secondary_tasks[key]

    print("==>selected {task} as other task".format(task=other_task))

    # get the label2idx
    train_fn = other_task.replace("dev", "train")
    _, _, secondary_label2idx = dutils.get_datasets([train_fn], tokenizer)

    # ===========================================================
    # check whether the labels correspond to the given label2idx or not
    # create data loader for each task
    primary_data, _, _ = dutils.get_datasets([primary_data_path], tokenizer, primary_label2idx)
    primary_is_multilabel = False

    # create batch loaders with the wrong index
    primary_data_loader = SwappableBatchGenerator(primary_data, batch_size, DEVICE, primary_is_multilabel,
                                                  len(primary_label2idx), 0)

    # predict
    print("checking label correspondence....")
    primary_golds, primary_preds = predict_with_gold(model, primary_data_loader)

    p = pearsonr(primary_golds, primary_preds)[0]
    print(p)
    # ===========================================================

    # create data loader for each task
    primary_data, _, _ = dutils.get_datasets([primary_data_path], tokenizer, primary_label2idx)
    secondary_data, _, _ = dutils.get_datasets([other_task], tokenizer, secondary_label2idx)
    secondary_label2idx = secondary_label2idx[0][0]

    primary_is_multilabel = False
    secondary_is_multilabel = dutils.sniff_multilabel(secondary_data)[0][0]

    print("==>secondary task is{n}multilabel".format(n=" " if secondary_is_multilabel else " not "))

    # create batch loaders with the wrong index
    primary_data_loader = SwappableBatchGenerator(primary_data, batch_size, DEVICE, primary_is_multilabel,
                                                  len(primary_label2idx), 1)
    secondary_data_loader = SwappableBatchGenerator(secondary_data, batch_size, DEVICE, secondary_is_multilabel,
                                                    len(secondary_label2idx), 0)

    # predict
    print("processing primary task data....")
    primary_golds, primary_preds = predict_with_gold(model, primary_data_loader)

    print("processing secondary task data....")
    secondary_golds, secondary_preds = predict_with_gold(model, secondary_data_loader)

    secondary_idx2label = {secondary_label2idx[i]: i for i in secondary_label2idx}
    save_predictions(primary_data_path, primary_preds, secondary_idx2label, model_path)

    # get correlation with gold labels
    # if the secondary task is single-label (Vent), compute eta (correlation ratio)
    # if it is multi-label, compute a correlation matrix R2 = cT * Rxx-1 * c, where c is the correlation vector between
    # Dreaddit and emotion and Rxx is the correlation matrix of the emotion labels among themselves
    # the correlation inside the matrix can be pearson's r
    if not secondary_is_multilabel:
        # correlation ratio between predictions and golds for the primary task
        eta, per_class = correlation_ratio(np.array(primary_golds), np.array(primary_preds))
        print("Correlation of gold stress and predicted emotion labels on Dreaddit:")
        print(eta)
        print("And per class...")
        print(per_class)
        print(secondary_label2idx)

        # and again for the inverse task!
        eta, per_class = correlation_ratio(np.array(secondary_preds), np.array(secondary_golds))
        print("Correlation of predicted stress and gold emotion labels on emotion data:")
        print(eta)
        print("And per class...")
        if p < 0:
            per_class = 1 - per_class
        print(per_class)
        print(secondary_label2idx)
    else:
        # multiple correlation coefficient between predictions and golds for the primary task
        R2, per_class = multiple_correlation_coefficient(np.array(primary_golds), np.array(primary_preds))
        print("Correlation of gold stress and predicted emotion labels on Dreaddit:")
        print(R2)
        print("And per class...")
        print(per_class)
        print(secondary_label2idx)

        # and again for the inverse task!
        R2, per_class = multiple_correlation_coefficient(np.array(secondary_preds), np.array(secondary_golds))
        print("Correlation of predicted stress and gold emotion labels on emotion data:")
        print(R2)
        print("And per class...")
        if p < 0:
            per_class = per_class * -1
        print(per_class)
        print(secondary_label2idx)


def do_true_task_swapping(model, data_path, tokenizer, labels2idxes, batch_size=16):
    """
    Perform multitask task-swapping analysis, full version
    @param model: the model to use
    @param data_path: the filename of the task data (just one at a time)
    @param tokenizer: a huggingface tokenizer or equivalent callable
    @param labels2idxes: the label2idx dictionaries for all tasks
    @param batch_size: the batch size, int
    """
    secondary_label2idx = labels2idxes[0][1]

    data, _, _, = dutils.get_datasets([data_path], tokenizer, labels2idxes)
    is_multilabel = dutils.sniff_multilabel(data)[0]

    data_loader = butils.SimultaneousBatchGenerator(data[0], batch_size, DEVICE, is_multilabel,
                                                    [len(l2i) for l2i in labels2idxes[0]])

    # predict
    print("processing primary task data....")
    golds, preds = predict_with_gold(model, data_loader, multi_flag=True)

    # i bet there's an unzip that does this but this works too
    # stress_preds = list(chain(*[p[0] for p in preds]))
    emo_preds = list(chain(*[p[1].tolist() for p in preds]))

    # get correlation with gold labels
    # if the secondary task is single-label (Vent), compute eta (correlation ratio)
    # if it is multi-label, compute a correlation matrix R2 = cT * Rxx-1 * c, where c is the correlation vector between
    # Dreaddit and emotion and Rxx is the correlation matrix of the emotion labels among themselves
    # the correlation inside the matrix can be pearson's r
    if not is_multilabel[1]:
        # correlation ratio between predictions and golds for the primary task
        eta, per_class = correlation_ratio(np.array(golds), np.array(emo_preds))
        print("Correlation of gold stress and predicted emotion labels on Dreaddit:")
        print(eta)
        print("And per class...")
        print(per_class)
        print(secondary_label2idx)
    else:
        # multiple correlation coefficient between predictions and golds for the primary task
        R2, per_class = multiple_correlation_coefficient(np.array(golds), np.array(emo_preds))
        print("Correlation of gold stress and predicted emotion labels on Dreaddit:")
        print(R2)
        print("And per class...")
        print(per_class)
        print(secondary_label2idx)


def interpret(model_path, test_file):
    """
    Run full model analysis
    @param model_path: the path of the model to use
    @param test_file: the path of the data to test on (the dev data, probably)
    """
    with open(model_path + "-meta.pkl", "rb") as f:
        meta = pickle.load(f)

    # load test data
    with open(test_file, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    # also load the label2idx; it should always be the same, but just in case
    try:
        label2idx = meta["label2idx"]
    except KeyError:
        label2idx = {"notstress": 0, "stress": 1}
    idx2label = {value: key for key, value in label2idx.items()}

    # create the model
    model = torch.load(model_path + "-params.pth", map_location="cpu")
    model.eval()

    logging.info("data loaded!")

    logging.info("*" * 20)
    logging.info("LIME RESULTS")

    # create tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, do_basic_tokenize=True)

    # LIME
    do_lime(model, model_path, data, tokenizer, idx2label)

    # task swapping
    if isinstance(model, BertMultitask):
        do_task_swapping(model, model_path, test_file, tokenizer, [[label2idx]])
    elif isinstance(model, SimultaneousMultitask):
        do_true_task_swapping(model, test_file, tokenizer, meta["labels2idxes"])


if __name__ == '__main__':
    # Change model name(s) here
    test_data = "data/dreaddit/dreaddit_dev.csv"
    for model_name in ['dreaddit_multitask_vent']:
        # ['dreaddit_bert', 'dreaddit_dreaddit-lm_bert', 'dreaddit_from_goemo-a_bert',
        # 'dreaddit_from_goemo-e_bert', 'dreaddit_from_goemo-s_bert', 'dreaddit_from_goemo-v_bert',
        # 'dreaddit_goemotions-a_multitask', 'dreaddit_goemotions-e_multitask', 'dreaddit_goemotions-s_multitask',
        # 'dreaddit_goemotions-v_multitask', 'dreaddit_goemotions-lm_bert', 'dreaddit_rnn',
        # 'dreaddit_vent_multitask', 'dreaddit_simmulti_goemos', 'dreaddit_simmulti_goemov']:

        # handle the model
        logging.info("evaluating {mod}...".format(mod=model_name))
        interpret("saved_models/" + model_name, test_data)
        logging.info("=" * 40)
