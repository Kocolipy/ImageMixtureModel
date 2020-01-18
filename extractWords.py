import json
import pickle
import pathlib
import pandas as pd
import os
import numpy as np
import utils

dir = pathlib.Path(r"D:\PML\Data\features")
outdir = pathlib.Path(r"D:\PML\Data\activation")

ACTIVATION_LEVELS = [0.3, 0.1]

for ACTIVATION_LEVEL in ACTIVATION_LEVELS:
    (outdir/(str(ACTIVATION_LEVEL) + "r")).mkdir(parents=True, exist_ok=True)
    documents = os.listdir(dir)
    for i, doc in enumerate(documents):
        if os.path.exists(str(outdir/(str(ACTIVATION_LEVEL) + "r")/(doc[:-4] + ".act"))):
            continue
        if (i + 1) % int(len(documents)/100) == 0:
            print(str((i + 1) * 100 / float(len(documents))) + "% completed!")
        output = np.array(utils.loadFeaturesF(dir/doc))
        output = [b.mean() for b in output.reshape(512, 49)]

        words = []
        for j in range(len(output)):
            if output[j] >= ACTIVATION_LEVEL:
                for _ in range(int(output[j]/ACTIVATION_LEVEL)):
                    words.append(j)

        dataFrame = pd.DataFrame(data={"words": words})
        dataFrame.to_feather(str(outdir/(str(ACTIVATION_LEVEL) + "r")/(doc[:-4] + ".act")))

