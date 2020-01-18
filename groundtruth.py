import pathlib
import json

photojson = pathlib.Path("D:\PML\\photo.json")

with open(photojson, "r") as f:
    lines = f.readlines()

labels = {"inside" : [], "outside" : [], "food" : [], "menu":[], "drink":[]}
for line in lines:
    obj = json.loads(line)
    labels[obj["label"]].append(obj["photo_id"] + ".jpg")

with open(pathlib.Path("D:\PML\\groundtruth.json"), "w+") as f:
    json.dump(labels, f)