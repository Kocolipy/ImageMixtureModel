import json
import numpy as np
import os
import pathlib
from sklearn.model_selection import StratifiedKFold

photos = []
labels = []

for subfolder in os.listdir("D:\PML\Data\photos"):
    new_photos = os.listdir("D:\PML\Data\photos\\" + subfolder)
    photos += new_photos
    labels += [subfolder]*len(new_photos)

photos = np.array(photos)
labels = np.array(labels)

skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(photos, labels)

def labels_split(arr):
    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, inverse, counts = np.unique(arr,
        return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes

def generate_folders(dirpath, photos, labels):
    dirpath.mkdir(parents=True)
    uniqueLabels, indices = labels_split(labels)

    for i in range(len(uniqueLabels)):
        selectPhotos = photos[indices[i]]
        select = []
        for p in selectPhotos:
            select.append(p)
        print(len(select), uniqueLabels[i])
        with open(dirpath/uniqueLabels[i], "w+") as f:
            json.dump(select, f)

for i, (train_index, test_index) in enumerate(skf.split(photos, labels)):
    X_train, X_test = photos[train_index], photos[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    dirpath = pathlib.Path("D:\PML\Data\\folds")
    generate_folders(dirpath/ str(i) / "train", X_train, y_train)
    print("Fold %d Training Set Generated" % i)
    generate_folders(dirpath/ str(i) / "val", X_test, y_test)
    print("Fold %d Test Set Generated" % i)




