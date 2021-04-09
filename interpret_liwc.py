from collections import defaultdict
import csv

from nltk import word_tokenize

import pandas as pd

# Change model names here
# models = ['dreaddit_bert', 'dreaddit_dreaddit-lm_bert', 'dreaddit_from_goemo-a_bert',
#           'dreaddit_from_goemo-e_bert', 'dreaddit_from_goemo-s_bert', 'dreaddit_from_goemo-v_bert',
#           'dreaddit_goemotions-a_multitask', 'dreaddit_goemotions-e_multitask', 'dreaddit_goemotions-s_multitask',
#           'dreaddit_goemotions-v_multitask', 'dreaddit_goemotions-lm_bert', 'dreaddit_vent_multitask']
# models = ['dreaddit_simmulti_goemos', 'dreaddit_simmulti_goemov']
models = ["dreaddit_multitask_vent"]

stress_unigrams = ['i', 'my', 'me', 'do', 'and', "'m", "n't", 'just', '’', 'feel', 'because', 'like', 'am', 'what',
                   'even', '?', 'he', 'but', 'anxiety', 'm', 'so', 'myself', 'this', 'know', 'ca', 'it', 'now', 'have',
                   'out', 'get', 'no', 'about', 't', 'feeling', 'up', 'bad', 'how', "'ve", 'scared', 'not', 'him',
                   'over', 'going', 'all', 'tell', 'right', 'stop', 'want', 'anxious', 'past', 'to', 'fucking', 'need',
                   'hate', 's', 'really', 'why', 'panic', 'where', 'happened', 'trying', 'still', 'when', 'days',
                   'makes', 'job', 'tired', 'or', 'shit', 'hard', 'getting', 'day', 'life', 'nothing', 'tl', 'dr',
                   'afraid', 'has', 'sorry', 'boyfriend', 'felt', 'crying', 'school', 'worse', 'don', 'go', 'attacks',
                   'sick', 'leave', 'deal', 'attack', 'anymore', 'being', 'work', 'im', 'having', 'constantly',
                   'thinking', 'almost', 'feels', 'been', 'worried', 'is', 'stress', 'which', 'family', 'due', 'fear',
                   'something', 'keep', 'everything', 'enough', 'every', 'back', 'worst', '...', 'point', 'home',
                   'sometimes', 'car', 'down', 'making', 'angry', 'literally', 'feelings', 'actually', 'cry',
                   'horrible', 'wo', 'think', 'anyone', 'end', 'move', '..', 'help', 'terrified', 'fuck', 'head',
                   'then', 'pain', 'losing', 'situation', 'depression', 'depressed', 've', 'made', 'money', 'coming',
                   'mom', 'safe', 'else', 'everyday', 'gets', 'honestly', 'thing', 'unable', 'turn', 'whole',
                   'terrible', 'alone', 'room', 'heart', 'saying', 'wake', 'awful', 'sleep', 'against', 'mentally',
                   'come', 'absolutely', 'nightmares', 'stupid', 'remember', 'lot', 'without', 'does', 'abuse', 'lose',
                   'class', 'sad', 'stuck', 'hell', 'suffer', 'cant', 'severe', 'emotions', 'leaving', '/',
                   'flashbacks', 'hospital', 'close', 'memories', 'off', 'night', 'nowhere', 'abused', 'knowing',
                   'issues', 'trigger', 'sexually']

nostress_uni = [',', 'you', 'the', 'a', 'her', 'she', 'we', 'for', '.', 'in', 'your', 'would', 'be', ')', '!', 'will',
                '(', '*', ':', '<', 'that', 'are', 'who', '>', 'as', 'was', 'url', 'more', 'if', 'years', '-', 'first',
                'were', 'their', 'thank', 'us', 'met', 'people', 'his', 'them', 'our', 'an', 'they', 'said', 'one',
                'together', 'others', 'share', 'let', 'best', 'food', 'other', '&', 'person', 'interested', 'please',
                'study', 'each', 'here', 'asked', 'link', 'treatment', 'those', 'free', 'could', '”', 'take', 'great',
                'same', 'support', 'good', '“', '[', 'some', 'make', 'months', 'may', 'older', 'finally', 'bit',
                'research', 'online', 'experience', 'little', 'through', 'hope', '#', '$', 'many', 'helped', 'edit',
                'decided', 'friend', 'see', 'took', 'few', 'homeless', 'wanted', 'nice', 'information', 'thanks',
                'around', "''", 'questions', 'any', 'date', 'went', 'later', 'everyone', 'looking', 'guys', 'ask',
                'than', 'relationship', 'ago', "'ll", 'sister', 'post', 'complete', "'d", 'dating', 'year', 'both',
                'current', 'mental', "'s", 'send', '18', 'moved', 'amazing', 'community', 'provide', 'items', 'read',
                'however', 'name', 'x200b', 'world', 'willing', 'different', 'guy', '3', 'turned', 'area', 'visit',
                'health', 'open', 'well', 'case', 'survivors', '10', 'hear', "'re", 'give', 'university', 'own', ']',
                'hi', 'learn', 'couple', 'access', 'old', 'long', 'eventually', 'choose', 'agreed', 'began', 'love',
                'reading', 'stories', 'loving', 'hey', 'experiences', 'include', 'preferences', 'forward', ';',
                'write', 'sub', '1', 'posted', 'also', 'loved', 'page', 'email', 'start', 'away', 'sleeping', 'note',
                'app', 'liked', 'helping', 'seemed', 'grateful', 'background', 'girl', 'talked', 'based', 'amazon', '2']


def load_csv(fn, tsv=False):
    """
    Load a csv from file
    @param fn: the name of the file
    @param tsv: Boolean, true if this is actually a tsv
    @return: data, a list of the csv's contents
    """
    with open(fn, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f, delimiter="\t" if tsv else ","))

    return data


def check_for_word_in_liwc(target_word, liwc_list):
    """
    Look for a given word in a given LIWC word list (may contain *)
    @param target_word: the target word
    @param liwc_list: the LIWC list
    @return: true if the word was found, false otherwise
    """
    liwc_words = liwc_list.split(" ")

    found = False

    for word in liwc_words:
        if word == target_word or ("*" in word and target_word.startswith(word[:-1])):
            found = True
            break

    return found


def do_liwc_analysis_csv(liwc, out_path):
    """
    Do all LIWC analysis from rationales for all models from a .csv
    @param liwc: the liwc dictionary, loaded by load_csv
    @param out_path: the path to write the final analysis to as a .csv
    """
    # now calculate how many rationales from each system are in what LIWC categories
    liwc_analysis = pd.DataFrame(0, index=[l["Category"] for l in liwc], columns=models)

    for model in models:
        # load rationales from a .csv
        print("{model}....".format(model=model))
        rationales = load_csv("lime/{f}-lime.csv".format(f=model))

        for i, datapoint in enumerate(rationales):
            # add each rationale to the correct LIWC category
            for word in datapoint["explanation"].split():
                for liwc_list in liwc:
                    if check_for_word_in_liwc(word, liwc_list["Words"]):
                        liwc_analysis[model][liwc_list["Category"]] += 1

    liwc_analysis.to_csv(out_path)


def do_liwc_analysis_txt(liwc, file_pattern, out_path):
    """
    Do all LIWC analysis from rationales for all models from a .txt
    @param liwc: the liwc dictionary, loaded by load_csv
    @param file_pattern: the format string for the file name
    @param out_path: the path to write the final analysis to as a .csv
    """
    # now calculate how many rationales from each system are in what LIWC categories
    liwc_analysis = pd.DataFrame(0, index=[l["Category"] for l in liwc], columns=models)

    for model in models:
        print("{model}....".format(model=model))
        # load rationales from a .txt
        with open(file_pattern.format(f=model), "r", encoding="utf-8") as f:
            rationales = [w.strip() for w in f.read().split("\n")]

        for word in rationales:
            for liwc_list in liwc:
                if check_for_word_in_liwc(word, liwc_list["Words"]):
                    liwc_analysis[model][liwc_list["Category"]] += 1

    liwc_analysis.to_csv(out_path)


def do_relative_salience_analysis(out_path):
    """
    Exactly like LIME from .csv, but with relative salience
    @param out_path: the file to write the results to (as a .csv)
    """
    relsal_analysis = pd.DataFrame(0, index=["stress", "nostress"], columns=models)

    for model in models:
        print("{model}....".format(model=model))
        rationales = load_csv("analysis/{f}-lime.csv".format(f=model))

        for i, datapoint in enumerate(rationales):
            # check rationales for relative salience words
            for word in datapoint["explanation"].split():
                if word in stress_unigrams:
                    relsal_analysis[model]["stress"] += 1
                if word in nostress_uni:
                    relsal_analysis[model]["nostress"] += 1

    relsal_analysis.to_csv(out_path)


def do_total_liwc_words(liwc, out_path):
    """
    Do LIWC analysis from total rationales file
    @param liwc: the liwc dictionary, loaded by load_csv
    @param out_path: the path to write the final analysis to as a .csv
    """
    # now calculate how many rationales from each system are in what LIWC categories
    liwc_analysis = pd.DataFrame(0, index=[l["Category"] for l in liwc], columns=models)

    for model in models:
        print("{model}....".format(model=model))
        rationales = load_csv("analysis/{f}-lime.csv".format(f=model))

        for i, datapoint in enumerate(rationales):
            counts = defaultdict(int)
            # add each rationale to the correct LIWC category
            for word in word_tokenize(datapoint["input"]):
                for liwc_list in liwc:
                    if counts[liwc_list["Category"]] < 10:
                        if check_for_word_in_liwc(word.lower(), liwc_list["Words"]):
                            liwc_analysis[model][liwc_list["Category"]] += 1
                            counts[liwc_list["Category"]] += 1

    liwc_analysis.to_csv(out_path)


def do_total_relsal_words(out_path):
    """
    Do relative salience analysis from total rationales file
    @param out_path: the file to write the results to (as a .csv)
    """
    relsal_analysis = pd.DataFrame(0, index=["stress", "nostress"], columns=models)

    for model in models:
        print("{model}....".format(model=model))
        rationales = load_csv("analysis/{f}-lime.csv".format(f=model))

        for i, datapoint in enumerate(rationales):
            counts = defaultdict(int)
            # check rationales for relative salience words
            for word in word_tokenize(datapoint["input"]):
                if counts["stress"] < 10:
                    if word.lower() in stress_unigrams:
                        relsal_analysis[model]["stress"] += 1
                        counts["stress"] += 1
                if counts["nostress"] < 10:
                    if word.lower() in nostress_uni:
                        relsal_analysis[model]["nostress"] += 1
                        counts["nostress"] += 1

    relsal_analysis.to_csv(out_path)


def main():
    """
    Examples of run commands can be found here
    """
    liwc = load_csv("data/liwc.tsv", tsv=True)

    # do_liwc_analysis_csv(liwc, "lime/all-rationales.csv")
    # do_liwc_analysis_txt(liwc, "lime/{f}-frequent-rationales.txt", "lime/frequent-rationales.csv")
    # do_liwc_analysis_txt(liwc, "lime/{f}-large-rationales.txt", "lime/large-rationales.csv")

    # also do this with relative salience unigrams
    # do_relative_salience_analysis("lime/relative_salience.csv")

    # do_liwc_analysis_csv(liwc, "lime/multi-all-rationales.csv")
    # do_relative_salience_analysis("lime/multi_relative_salience.csv")

    # do_liwc_analysis_csv(liwc, "lime/vent-all-rationales.csv")
    # do_relative_salience_analysis("lime/vent_relative_salience.csv")

    do_total_liwc_words(liwc, "lime/all-all-rationales.csv")
    do_total_relsal_words("lime/all_relative_salience.csv")


if __name__ == "__main__":
    main()

