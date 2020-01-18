import os
import json
import pathlib
import numpy as np
import pandas as pd
import feather


def groundtruth():
    groundtruthfile = pathlib.Path("D:\PML\\groundtruth.json")
    return loadJson(groundtruthfile)


def loadJson(filename):
    with open(str(filename), "r") as f:
        return json.load(f)


def loadDocuments(directory):
    documents = []
    for c in os.listdir(str(directory)):
        docs = loadJson(directory / c)
        documents += docs
    return np.array(documents)


def loadCheckpoint(ckptpath):
    dct = loadJson(ckptpath)
    return np.array(dct["z"]), np.array(dct["occurrences"]), np.array(dct["doc_count"]), np.array(dct["documents"])


def loadF(filepath):
    df = pd.read_feather(str(filepath), columns=None, use_threads=True);
    return df


def loadWordsF25088(filename, activation):
    # words_dir = pathlib.Path("D:\PML\Data\\activation\\{0}".format(activation))
    words_dir = pathlib.PureWindowsPath(r"C:\Users\Nicholas\Desktop\{0}".format(activation))
    return loadF(words_dir/(filename + ".act")).words.tolist()


def loadWordsF7x7(filename, activation):
    words = loadWordsF25088(filename, activation)
    return [int(i) for i in np.array(words) / 49]


def loadWordsMean(filename, activation):
    words_dir = pathlib.Path("D:\PML\Data\\activation\\{0}m".format(activation))
    return loadF(words_dir/(filename + ".act")).words.tolist()


def loadWordsR(filename, activation):
    words_dir = pathlib.Path("D:\PML\Data\\activation\\{0}r".format(activation))
    return loadF(words_dir/(filename + ".act")).words.tolist()


def loadWordsF(filename, activation):
    return loadWordsR(filename, activation)


def loadFeaturesF(filepath):
    df = loadF(filepath)
    return df.output.tolist()