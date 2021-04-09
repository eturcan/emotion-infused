# native python libraries
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import logging
import os
import pickle

# ML, linalg, stats
import numpy as np
import sklearn.metrics as skmetric

# pytorch
import torch
import torch.nn as nn

# transformers
from transformers import AdamW
import transformers.optimization as tfoptim

# hyperparameter optimization
import ax

# our files
from configs import const
from configs.defaults import DEFAULTS
import utils.batch_utils as butils
import utils.data_utils as dutils
import utils.loss_utils as lutils
import utils.meta_utils as meutils
import utils.model_utils as moutils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = const.SEEDS[0]


# ==================================================================
#     WORKHORSE FUNCTIONS (TRAIN, EVALUATE, PREDICT)
# ==================================================================

def predict(model, data_loader):
    """
    Run the given model on the given data and return its predictions
    @param model: an instance of one of our models (models/)
    @param data_loader: an instance of one of our batch generators (utils/batch_utils)
    @return: all_preds, a list of predictions (for one task)
    """

    model.eval()
    all_preds = []

    with torch.no_grad():
        # run through all batches in train generator
        for j, (dataset_id, tokens, token_types, attn_mask, _) in enumerate(data_loader.get_batches()):
            # get model predictions
            preds = model.predict(dataset_id, tokens, token_types, attn_mask)

            all_preds.extend(preds.tolist())

    return all_preds


def evaluate(model, data_loader, metrics):
    """
    Evaluate the model on the given data and return its scores
    @param model: an instance of one of our models (models/)
    @param data_loader: an instance of one of our batch generators (utils/batch_utils)
    @param metrics: a dictionary of callable metric functions that operate on the output of our models
    @return: calculated_metrics, a dictionary with the same keys as metrics but calculated numbers as the values
    """

    model.eval()
    all_golds = []
    all_preds = []

    with torch.no_grad():
        # run through all batches in train generator
        for j, (dataset_id, tokens, token_types, attn_mask, golds) in enumerate(data_loader.get_batches()):
            # get model predictions
            preds = model.predict(dataset_id, tokens, token_types, attn_mask)

            # if we have a simultaneous multitask model, ignore the secondary tasks
            # (this should never actually be a problem since we create dev/test loaders as singletask)
            if len(golds) > 1 and isinstance(golds[0], list):
                golds = golds[0]

            all_golds.extend(golds.tolist())
            all_preds.extend(preds.tolist())

    calculated_metrics = {}
    for metric_name in metrics:
        calculated_metrics[metric_name] = metrics[metric_name](all_golds, all_preds)

    return calculated_metrics


def train(model, train_loader, loss_calculator, optimizer, scheduler, dev_loader, dev_metrics, target_metric_name,
          kwargs):
    """
    Train the given model on the given data and return the trained model and its dev/test performance
    Includes early stopping
    @param model: an instance of one of our models (models/)
    @param train_loader: an instance of one of our data loaders (utils/data_utils)
    @param loss_calculator: an instacne of one of our loss calculators (utils/loss_utils)
    @param optimizer: a Transformers optimizer
    @param scheduler: a Transformers scheduler
    @param dev_loader: an instance of one of our data loaders (utils/data_utils)
    @param dev_metrics: a dictionary of callable metric functions that operate on the output of our models
    @param target_metric_name: one of the keys from dev_metrics that will be the optimization criterion
    @param kwargs: kwargs generated from this file's __main__
    """

    # track info needed for early stopping
    dev_scores = []
    best_dev_score = -np.inf
    best_epoch = 0
    epochs_wo_improvement = 0

    for i in range(kwargs.epochs):
        logging.info("=====EPOCH {i}=====".format(i=i))
        model.train()

        # run through all batches in train generator
        for j, (dataset_id, tokens, token_types, attn_mask, golds) in enumerate(train_loader.get_batches()):
            # get model predictions
            preds, loss = model.get_loss(dataset_id, tokens, token_types, attn_mask, loss_calculator, golds)

            # backward pass, clip gradients, optimizer step
            optimizer.zero_grad()
            loss.backward()
            if kwargs.clip_value > 0:
                nn.utils.clip_grad_norm_(model.parameters(), kwargs.clip_value)
            optimizer.step()
            scheduler.step()

            # log info
            if j % kwargs.print_every == 0:
                logging.info("[Epoch {i}] [Batch {j}/{batches}] loss={loss:.4f}".format(
                    i=i, j=j, batches=len(train_loader), loss=loss
                ))

        # get dev performance
        dev_results = evaluate(model, dev_loader, dev_metrics)

        logging.info("DEV RESULTS FOR EPOCH {i}:".format(i=i))
        for metric in dev_results:
            logging.info("{metric_name}: {metric_value:.4f}".format(metric_name=metric,
                                                                    metric_value=dev_results[metric]))

        # need to check if no improvement >= tolerance for patience epochs
        # if this is the best epoch, save its parameters
        dev_scores.append(dev_results[target_metric_name])

        # if this is the strictly best epoch (or the first epoch), save its params
        # note: this "best score" check bypasses the tolerance requirement
        if len(dev_scores) == 1 or dev_scores[-1] > max(dev_scores[:-1]):
            best_epoch = i
            best_dev_score = dev_scores[-1]
            epochs_wo_improvement = 0
            torch.save(model.state_dict(), kwargs.tmp_fn)
        elif dev_scores[-1] >= dev_scores[-2] + kwargs.tolerance:
            # otherwise, if at least improving, reset the patience
            epochs_wo_improvement = 0
        else:
            # if not improving, increment the patience
            epochs_wo_improvement += 1
            logging.info("No improvement for {j} epoch{s}....".format(j=epochs_wo_improvement,
                                                                      s="" if epochs_wo_improvement == 1 else "s"))
            # if we have reached the patience, stop training
            if epochs_wo_improvement >= kwargs.patience:
                logging.info("Patience exceeded at epoch {i}".format(i=i))
                break

    # before stopping, load the best set of parameters if they were not the last set
    # then delete the temp file
    if best_epoch != kwargs.epochs - 1:
        model.load_state_dict(torch.load(kwargs.tmp_fn))

    if os.path.exists(kwargs.tmp_fn):
        os.remove(kwargs.tmp_fn)

    logging.info("Best epoch was {j}, "
                 "with best {metric_name} {metric_value:.4f}".format(j=best_epoch,
                                                                     metric_name=target_metric_name,
                                                                     metric_value=best_dev_score))


# ==================================================================
#     MAIN TRAIN FUNCTION
# ==================================================================

def train_main(seed, args):
    """
    Train one model from scratch, including creating it and all its optimizers, loss functions, etc.
    @param seed: an integer random seed
    @param args: kwargs generated from this file's __main__
    @return:
    """
    # random filename for saving params -- get BEFORE setting random seed
    if args.tmp_fn is None:
        args.tmp_fn = "params-{n}.tmp".format(n=np.random.randint(10000000, 99999999))

    logging.info("Saving temporary parameters to {fn}....".format(fn=args.tmp_fn))

    # set real random seed upon starting training
    meutils.set_random_seed(seed)

    logging.info("%" * 40)
    logging.info("NEW TRAINING RUN")
    logging.info("Random seed: {seed}".format(seed=seed))
    logging.info(args)
    logging.info("%" * 40)

    # for the classical multitask model
    if args.model == "multi":
        simultaneous_multi = True
    else:
        simultaneous_multi = False

    logging.info("Loading data....")
    all_datasets = [args.main_dataset] if args.aux_datasets is None else [args.main_dataset] + args.aux_datasets

    # create data and batcher
    # _datas will be a list of lists of tuples
    # one outer list for the datasets
    # one inner list for the tuples in that dataset
    # inside the inner list, each element is a tuple of (token_ids, type_ids, attention_mask, label1, label2, ...)
    # labels2idxes is a list of lists of dictionaries (outer list=datasets, inner list=tasks, dict=labels)
    train_datas, tokenizer, labels2idxes = dutils.get_datasets(all_datasets)
    dev_data, _, _ = dutils.get_datasets([args.dev_file], tokenizer=tokenizer, label2idx=[labels2idxes[0]])
    test_data, _, _ = dutils.get_datasets([args.test_file], tokenizer=tokenizer, label2idx=[labels2idxes[0]])

    # a lists of lists of booleans
    # one outer list the datasets, one inner list for the tasks in that dataset
    # most commonly the inner lists will be singletons
    train_is_multilabel = dutils.sniff_multilabel(train_datas)

    args.tokenizer = tokenizer

    logging.info("Creating model....")
    # create model
    model, task_setting = moutils.create_model(labels2idxes, train_is_multilabel, args)
    model = model.to(DEVICE)

    # training can be multitask; evaluation will not

    # create dataloaders (can be useful for multitask)
    if simultaneous_multi:
        train_loader = butils.SimultaneousBatchGenerator(train_datas[0], args.batch_size, DEVICE,
                                                         train_is_multilabel[0], [len(l2i) for l2i in labels2idxes[0]])

        # dev and eval will operate on only the first task (stress), so they are the same as for non-multitask
    else:
        if task_setting == "single":
            # we care only about one dataset and one task
            train_loader = butils.SimpleBatchGenerator(train_datas, args.batch_size, DEVICE, train_is_multilabel[0][0],
                                                       len(labels2idxes[0][0]))
        else:
            # we may care about multiple datasets but we care about only one task
            train_loader = butils.RoundRobinBatchGenerator(train_datas, args.batch_size, DEVICE, [tm[0] for tm in
                                                                                                  train_is_multilabel],
                                                           [len(l2i[0]) for l2i in labels2idxes])

    dev_loader = butils.SimpleBatchGenerator(dev_data, args.batch_size, DEVICE, train_is_multilabel[0][0],
                                             len(labels2idxes[0][0]), shuffle=False)
    test_loader = butils.SimpleBatchGenerator(test_data, args.batch_size, DEVICE, train_is_multilabel[0][0],
                                              len(labels2idxes[0][0]), shuffle=False)

    # create loss functions
    if simultaneous_multi:
        loss_functions = lutils.get_simultaneous_loss_functions(train_datas, train_is_multilabel[0], DEVICE, args)
        loss_calculator = lutils.SimultaneousLossCalculator(loss_functions, args.stress_weight)
    else:
        loss_functions = lutils.get_loss_functions(train_datas, [tm[0] for tm in train_is_multilabel], DEVICE, args)
        loss_calculator = lutils.LossCalculator(loss_functions)

    # create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # create scheduler for optimizer
    scheduler = tfoptim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        (len(train_loader) * args.epochs) / 10), num_training_steps=len(train_loader) * args.epochs)

    # create evaluation metrics
    # note that 'binary' could theoretically cause a problem if we had a two-class multilabel task
    dev_metrics = {
        "dev accuracy": lambda gold, pred: skmetric.accuracy_score(gold, pred),
        "dev f1": lambda gold, pred: skmetric.f1_score(gold, pred,
                                                       average="binary" if len(labels2idxes[0]) == 2 else "macro")
    }

    eval_metrics = {
        "eval accuracy": lambda gold, pred: skmetric.accuracy_score(gold, pred),
        "eval f1": lambda gold, pred: skmetric.f1_score(gold, pred,
                                                        average="binary" if len(labels2idxes[0]) == 2 else "macro")
    }

    # NOTE: we want the hamming loss to be minimized, unlike the other metrics
    if train_is_multilabel[0][0]:
        dev_metrics["dev hamming"] = lambda gold, pred: skmetric.hamming_loss(gold, pred)
        eval_metrics["eval hamming"] = lambda gold, pred: skmetric.hamming_loss(gold, pred)

    target_metric = "dev f1"

    # train
    logging.info("Training model....")
    train(model, train_loader, loss_calculator, optimizer, scheduler, dev_loader, dev_metrics, target_metric, args)

    logging.info("Collecting final dev metrics for serialization....")
    dev_results = evaluate(model, dev_loader, dev_metrics)

    # final evaluations
    logging.info("FINAL DEV RESULTS:")
    for metric in dev_results:
        logging.info("{metric_name}: {metric_value:.4f}".format(metric_name=metric,
                                                                metric_value=dev_results[metric]))

    logging.info("Evaluating model on test set....")
    eval_results = evaluate(model, test_loader, eval_metrics)

    logging.info("EVAL RESULTS:")
    for metric in eval_results:
        logging.info("{metric_name}: {metric_value:.4f}".format(metric_name=metric,
                                                                metric_value=eval_results[metric]))

    # return model, results, and meta-info
    meta = {"tokenizer": tokenizer,
            "labels2idxes": labels2idxes,
            "is_multilabel": train_is_multilabel,
            "args": args,
            "dev_results": dev_results,
            "eval_results": eval_results}

    return model, meta


# ==================================================================
#     OPTIMIZATION & SETUP
# ==================================================================

def optimize(parameters, kwargs):
    """
    Optimize a given model type using ax, running the model multiple times
    @param parameters: a dictionary of parameters to optimize and their bounds and types, required by ax
    @param kwargs: kwargs generated by this file's __main__
    """
    def get_score_for_parameters(parameters, kwargs):
        """
        Run one model and get its metrics of interest
        @param parameters:
        @param kwargs:
        @return:
        """
        # change the kwargs according to the given parameterization
        for param_name in parameters:
            setattr(kwargs, param_name, parameters[param_name])

        avg_dev = defaultdict(float)
        avg_eval = defaultdict(float)

        # optimization results based on an average of n runs with distinct random seeds
        for i in range(kwargs.num_restarts):
            _, meta = train_main(const.SEEDS[i], kwargs)

            # add up the results from each run
            for collector, key in zip([avg_dev, avg_eval], ["dev_results", "eval_results"]):
                for metric_name in meta[key]:
                    # we know ahead of time how many runs we need, so do the averaging as we go (/ num_restarts)
                    collector[metric_name] += (meta[key][metric_name] / kwargs.num_restarts)

        return {"dev f1": (avg_dev["dev f1"], 0),
                "test f1": (avg_eval["eval f1"], 0),
                "dev accuracy": (avg_dev["dev accuracy"], 0),
                "test accuracy": (avg_eval["eval accuracy"], 0)}

    best_parameters, values, experiment, best_model = ax.service.managed_loop.optimize(
        parameters=parameters,
        evaluation_function=lambda params: get_score_for_parameters(params, kwargs),
        objective_name="dev f1",
        total_trials=kwargs.trials,
    )

    df = experiment.fetch_data().df

    # record final info about best parameter settings
    logging.info("=" * 30)
    logging.info("Finished optimizing!")
    logging.info("BEST PARAMETER SETTINGS...")
    for param in best_parameters:
        logging.info("{param_name}: {param_value}".format(param_name=param, param_value=best_parameters[param]))
    logging.info("BEST DEV SCORE:")
    logging.info(max(df[df["metric_name"] == "dev f1"]["mean"]))
    logging.info("=" * 30)

    print(best_parameters)

    # finally, if saving, train a single new model with the best settings and save it
    if kwargs.save_path is not None:
        logging.info("Saving one model with the best parameters....")

        for param_name in best_parameters:
            if param_name == "dropout":
                kwargs.dropout = best_parameters[param_name]
            elif param_name == "embed_dim":
                kwargs.embed_dim = best_parameters[param_name]
            if param_name == "hidden_dim":
                kwargs.hidden_dim = best_parameters[param_name]
            elif param_name == "lr":
                kwargs.lr = best_parameters[param_name]
            elif param_name == "num_layers":
                kwargs.num_layers = best_parameters[param_name]

        model, meta = train_main(const.SEEDS[0], kwargs)

        if kwargs.save_lm:
            logging.info("Saving only language model to {fn}....".format(fn=kwargs.save_path + "-lm"))
            model.bert.save_pretrained(kwargs.save_path + "-lm")
        else:
            logging.info("Saving entire model to {fn}....".format(fn=kwargs.save_path + "-params.pth"))
            logging.info("Saving meta info to {fn}....".format(fn=kwargs.save_path + "-meta.pkl"))
            torch.save(model, kwargs.save_path + "-params.pth")
            with open(kwargs.save_path + "-meta.pkl", "wb+") as f:
                pickle.dump(meta, f)


def train_setup(kwargs):
    """
    Run training in the requested way, either running 1+ random restarts or parameter optimization
    @param kwargs: kwargs generated from thie file's __main__
    """

    if kwargs.optimize:
        if kwargs.model == "bert" or kwargs.model == "multi_alt":
            parameters = [
                {"name": "lr", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
                {"name": "dropout", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"}
            ]
        elif kwargs.model == "multi":
            parameters = [
                {"name": "lr", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
                {"name": "dropout", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"},
                {"name": "stress_weight", "type": "range", "bounds": [0.0, 0.9], "value_type": "float"}
            ]
        else:
            raise ValueError("Don't know how to optimize that yet.")

        optimize(parameters, kwargs)
    else:
        # run model a given number of times and report average performance
        if kwargs.num_restarts == 1:
            model, meta = train_main(SEED, kwargs)

            # save model and all results if requested
            if kwargs.save_path is not None:
                if kwargs.save_lm:
                    logging.info("Saving only language model....")
                    model.bert.save_pretrained(kwargs.save_path + "-lm")
                else:
                    logging.info("Saving entire model and meta....")
                    torch.save(model, kwargs.save_path + "-params.pth")
                    with open(kwargs.save_path + "-meta.pkl", "wb+") as f:
                        pickle.dump(meta, f)
        else:
            if kwargs.save_lm:
                raise ValueError("Don't know how to save one language model from multiple random restarts...")

            # run the model n times and report average results
            all_dev_results = defaultdict(list)
            all_eval_results = defaultdict(list)
            all_models = []
            all_metas = []

            for i in range(kwargs.num_restarts):
                logging.info("--------------TRAINING #{i}--------------".format(i=i))

                this_model, this_meta = train_main(const.SEEDS[i], kwargs)

                # add metrics together
                all_models.append(this_model)
                all_metas.append(this_meta)

                for key, collector in zip(["dev_results", "eval_results"], [all_dev_results, all_eval_results]):
                    for metric in this_meta[key]:
                        collector[metric].append(this_meta[key][metric])

            stdev_dev = {}
            stdev_eval = {}

            avg_dev = {}
            avg_eval = {}

            for collector, avg, stdev in zip([all_dev_results, all_eval_results],
                                             [avg_dev, avg_eval],
                                             [stdev_dev, stdev_eval]):
                for metric in collector:
                    avg[metric] = np.mean(collector[metric])
                    stdev[metric] = np.std(collector[metric])

            final_meta = {"avg_dev_scores": avg_dev,
                          "stdev_dev_scores": stdev_dev,
                          "avg_eval_scores": avg_eval,
                          "stdev_eval_scores": stdev_eval,
                          "model_metas": all_metas}

            if kwargs.save_path is not None:
                logging.info("Saving all models and meta....")
                for i, model in enumerate(all_models):
                    torch.save(model, kwargs.save_path + "-" + str(i) + "-params.pth")
                with open(kwargs.save_path + "-meta.pkl", "wb+") as f:
                    pickle.dump(final_meta, f)


def predict_evaluate_setup(kwargs):
    """
    Predict, including loading model and data. Optionally run evaluation if we also have the gold
    @param kwargs: kwargs generated from this file's __main__
    """

    with open(kwargs.model_path + "-meta.pkl", "rb") as f:
        meta = pickle.load(f)

    # load data (create its own label2idx, just throw it away)
    logging.info("Loading data....")
    data, _, l2i = dutils.get_datasets([kwargs.data], meta["tokenizer"])

    # for prediction, we care only about the first task in the first dataset
    is_multilabel = meta["is_multilabel"]

    # label2idx, also for just one task
    idx2label = {value: key for key, value in meta["labels2idxes"].items()}

    # create dataloader
    data_loader = butils.SimpleBatchGenerator(data, meta["args"].batch_size, DEVICE, is_multilabel, len(l2i[0]),
                                              shuffle=False)

    # create the model
    logging.info("Loading model....")
    model = torch.load(kwargs.model_path + "-params.pth", map_location=DEVICE)

    # predict the labels
    logging.info("Predicting labels....")
    all_preds = predict(model, data_loader)

    # save the labels in text form
    labels = []
    for pred in all_preds:
        if isinstance(pred, list):
            labels.append(",".join([idx2label[i] for i, val in enumerate(pred) if val == 1]))
        else:
            labels.append(idx2label[pred])

    with open(kwargs.data, "r", encoding="utf-8") as f:
        input_data = list(csv.reader(f))[1:]

    with open(kwargs.out_path, "w+", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emotions"])

        for datapoint, this_labels in zip(input_data, labels):
            writer.writerow([datapoint[0], this_labels])

    logging.info("Finished! Labels saved to {fn}.".format(fn=kwargs.out_path))


# ==================================================================
#     ARGPARSE
# ==================================================================

if __name__ == "__main__":
    # track time
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="actions to perform")

    ########## TRAIN ARGUMENTS ##########
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train_setup, mode="train")

    data_group = train_parser.add_argument_group("data")
    data_group.add_argument("--main_dataset", type=str, required=True,
                            help="path to the main dataset. expected to be a .csv with headers: text, label.")
    data_group.add_argument("--aux_datasets", type=str, nargs="+",
                            help="space-separated list of paths to auxiliary datasets. "
                                 "expected to be same format as main.")
    data_group.add_argument("--dev_file", type=str, required=True,
                            help="path to the dev data. expected to be same format as main.")
    data_group.add_argument("--test_file", type=str, required=True,
                            help="path to the test data. expected to be same format as main.")

    model_group = train_parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, choices=["bert", "multi_alt", "multi"],
                             required=True,
                             help="the name of the model to use. some models may use different model args than others.")
    model_group.add_argument("--encoder", type=str, choices=["lstm", "gru"], default=DEFAULTS["encoder"],
                             help="the type of recurrent layer to use for the RNN model. ignored for others.")
    model_group.add_argument("--embed_dim", type=int, default=DEFAULTS["embed_dim"],
                             help="size of the embeddings used by the model. ignored for BERT.")
    model_group.add_argument("--hidden_dim", type=int, default=DEFAULTS["hidden_dim"],
                             help="size of the model's hidden layer. ignored for BERT.")
    model_group.add_argument("--num_layers", type=int, default=DEFAULTS["num_layers"],
                             help="number of RNN of transformer layers to use. ignored for BERT.")
    model_group.add_argument("--dropout", type=float, default=DEFAULTS["dropout"],
                             help="dropout to apply to the model during training")
    model_group.add_argument("--bert", type=str, default=DEFAULTS["bert"],
                             help="a path to a pretrained BERT model, or the name of such a model supported by "
                                  "huggingface (e.g., default: 'bert-base-uncased')")

    train_group = train_parser.add_argument_group("train")
    train_group.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                             help="number of epochs to train (max epochs, if early stopping).")
    train_group.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                             help="number of epochs to wait for validation improvement before exiting.")
    train_group.add_argument("--tolerance", type=float, default=DEFAULTS["tolerance"],
                             help="amount the validation performance must improve to be considered \"improving\".")
    train_group.add_argument("--no_early_stop", action="store_true",
                             help="include this flag to turn off early stopping.")
    train_group.add_argument("--main_only_epochs", type=int, default=DEFAULTS["main_only_epochs"],
                             help="number of epochs to train on only the main dataset after the initial training. "
                                  "if 0, will not train further.")
    train_group.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"],
                             help="the size of the batches to use when training and evaluating.")
    train_group.add_argument("--lr", type=float, default=DEFAULTS["lr"],
                             help="initial learning rate for the optimizer.")
    train_group.add_argument("--clip_value", type=float, default=DEFAULTS["clip_value"],
                             help="the value at which to clip gradients. -1 for no gradient clipping.")
    train_group.add_argument("--class_weights", action="store_true",
                             help="include this flag to use class weights in the loss calculations.")
    train_group.add_argument("--stress_weight", type=float, default=DEFAULTS["stress_weight"],
                             help="weight of the stress task for the Multi model. emotion is weighted with 1 - this.")

    meta_group = train_parser.add_argument_group("meta")
    meta_group.add_argument("--tmp_fn", help="an ID for the para meter checkpoints. will be randomly generated on the "
                                             "order of 10^7 if not given. checkpoints will be temporarily saved as "
                                             "'params-{tmp_fn}.tmp'.")
    meta_group.add_argument("--optimize", action="store_true",
                            help="include this flag to use the ax library to tune parameters.")
    meta_group.add_argument("--trials", type=int, default=DEFAULTS["trials"],
                            help="number of hyperparameter trials to attempt. ignored if not optimizing.")
    meta_group.add_argument("--num_restarts", type=int, default=DEFAULTS["num_restarts"],
                            help="the number of random restarts to average. if optimizing, run each parameter setting "
                                 "this many times. we have 10 random seeds predefined in configs/const.py; more "
                                 "restarts than this will cause an error unless you add more seeds.")
    meta_group.add_argument("--save_path", type=str, default=DEFAULTS["save_path"],
                            help="the path to save parameters and metadata. we will append -params.pth or -meta.pkl to "
                                 "this string to save the data.")
    meta_group.add_argument("--save_lm", action="store_true",
                            help="include this flag to save only the language model and discard classification layers "
                                 "(good for fine-tuning)")
    meta_group.add_argument("--print_every", type=int, default=DEFAULTS["print_every"],
                            help="log training info every so many batches.")
    meta_group.add_argument("--log", action="store_true")
    meta_group.add_argument("--logfile", type=str,
                            default="{currenttime}.log".format(currenttime=datetime.now().strftime("%m%d_%H:%M:%S")))

    ########## PREDICT ARGUMENTS ##########
    predict_parser = subparsers.add_parser("predict")
    predict_parser.set_defaults(func=predict_evaluate_setup, mode="predict")

    predict_data_group = predict_parser.add_argument_group("data")
    predict_data_group.add_argument("--data", type=str, required=True,
                                    help="path to the data for prediction. expected to be a .csv with headers: text, "
                                         "(anything else).")
    predict_data_group.add_argument("--model_path", type=str, required=True,
                                    help="path to a trained model, not a language model. stop before .params and "
                                         "-meta.pkl.")
    predict_data_group.add_argument("--out_path", type=str, required=True,
                                    help="path to store the predictions.")

    predict_meta_group = predict_parser.add_argument_group("meta")
    predict_meta_group.add_argument("--log", action="store_true")
    predict_meta_group.add_argument("--logfile", type=str,
                                    default="{currenttime}.log".format(currenttime=datetime.now().strftime("%m%d_%H:%M:%S")))

    ########## EVALUATE ARGUMENTS ##########
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.set_defaults(func=predict_evaluate_setup, model="evaluate")

    # parse arguments
    kwargs = parser.parse_args()

    # error checking -- throws errors if anything is wrong
    meutils.validate_args(kwargs)
    logging.info("Successfully validated arguments!")

    # set up logger
    logFormatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %I:%M:%S %p")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.pop()  # pop off the default handler for whatever reason....

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    if kwargs.log:
        fileHandler = logging.FileHandler(kwargs.logfile)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

    # run main -- predict, evaluate, or train
    kwargs.func(kwargs)

    # log total running time
    end_time = datetime.now()
    logger.info("====>Process took {t}.".format(t=str(end_time - start_time)))
