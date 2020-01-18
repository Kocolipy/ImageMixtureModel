import argparse
import json
import numpy as np
import os
import pathlib
import utils


def saveCkpt(z, occurrences, doc_count, documents, fp):
    dct = { "doc_count": doc_count.tolist(), "z": z.tolist(), "occurrences": occurrences.tolist(), "documents": documents.tolist()}
    json.dump(dct, fp)


def gibbs(W, α, γ, K, activation_level, ckptdir, ckpt=None):
    # documents are a list of all document names
    # ckptdir is the directory where checkpoints are stored
    # ckpt is the current checkpoint (int)

    # z[i] = class of document i, where i enumerates the distinct doc_labels
    # doc_count[k] = number of documents of class k
    if ckpt is None:
        documents = utils.loadDocuments(directory=training_dir)
        z = np.random.choice(K, len(documents))

        doc_count = np.zeros(K, dtype=int)
        # occurrences[k,w] = number of occurrences of word_id w in documents of class k
        # word_count[k] = total number of words in documents of class k
        occurrences = np.zeros((K, W))

        for i, d in enumerate(documents):
            if (i + 1) % int(len(documents)/250) == 0:
                print(str((i + 1) *100 / float(len(documents))) + "% completed!")
                print(doc_count)
            doc_count[z[i]] += 1
            w = utils.loadWordsF(d, activation_level)
            for word in w:
                occurrences[z[i], word] += 1

        with open(str(ckptdir/"0.ckpt"), "w+") as f:
            saveCkpt(z, occurrences, doc_count, documents, f)
        print("Initial Loading completed!")
        ckpt = 0

    else:
        z, occurrences, doc_count, documents = utils.loadCheckpoint(ckptdir/(str(ckpt) + ".ckpt"))
        print("Loaded from Checkpoint")

    word_count = np.sum(occurrences, axis=1)

    while True:
        ckpt += 1
        for i in range(len(documents)):
            if (i + 1) % int(len(documents)/100) == 0:
                print(str((i + 1) *100 / float(len(documents))) + "% completed!")
            # get the words,counts for document i
            # and remove this document from the counts
            w = utils.loadWordsF(documents[i], activation_level)

            for word in w:
                occurrences[z[i], word] -= 1
            word_count[z[i]] -= len(w)
            doc_count[z[i]] -= 1

            # Find the log probability that this document belongs to class k, marginalized over θ and β
            logp = []
            for k in range(K):
                value = 0
                probw = np.log(γ + occurrences[k]) - np.log(γ * W + word_count[k])
                for word in w:
                    value += probw[word]
                logp.append(value + np.log(α + doc_count[k]))
            p = np.exp(logp - np.max(logp))
            p = p/sum(p)

            # Assign this document to a new class, chosen randomly, and add back the counts
            k = np.random.choice(K, p=p)
            z[i] = k

            for word in w:
                occurrences[z[i], word] += 1

            word_count[k] += len(w)
            doc_count[k] += 1

        with open(str(ckptdir/(str(ckpt) + ".ckpt")), "w+") as f:
            saveCkpt(z, occurrences, doc_count, documents, f)
        print("Checkpoint", str(ckpt), "completed!")

        yield np.copy(z)


parser = argparse.ArgumentParser(description='Run the Gibbs Sampler.')
parser.add_argument('α', metavar='α', type=float, nargs='+',
                   help='the prior for the topic distribution')
parser.add_argument('γ', metavar='γ', type=float, nargs='+',
                   help='the prior for the word-topic distribution')
parser.add_argument('activation', metavar='activation level', type=float, nargs='+',
                   help='Activation Threshold of the features')
parser.add_argument('--fold', dest='fold', type=int, nargs='?',
                    help='The fold number for cross validation (Default: 0)')
parser.add_argument('--iteration', dest='iteration', type=int, nargs='?',
                    help='The number of gibbs sweeps to run (Default: 100)')
parser.add_argument('--ckpt', dest='ckpt', type=int, nargs='?',
                    help='Checkpoint to begin from')

args = parser.parse_args()
K=5
α=args.α[0]
γ=args.γ[0]
acti_level = args.activation[0]
current_fold = 0 if args.fold is None else args.fold

training_dir = pathlib.Path("D:\PML\Data\\folds\\{0}\\train".format(current_fold))
output_dir = pathlib.Path("D:\PML\Data\\repmean\\checkpoints\\a_{0}\\g_{1}\\act_{2}_{3}".format(α, γ, acti_level, current_fold))
output_dir.mkdir(parents=True, exist_ok=True)

g = gibbs(W=512, α=α, γ=γ, K=K, activation_level=acti_level, ckptdir=output_dir, ckpt=args.ckpt)
NUM_ITERATIONS = 50 if args.iteration is None else args.iteration
for i in range(NUM_ITERATIONS):
    z = next(g)
