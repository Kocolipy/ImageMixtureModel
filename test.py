import pathlib
import os
import mpmath as mp
import numpy as np
import json
import utils
import matplotlib.pyplot as plt

def getModel(ckptdir, convergence):
    ### convergence is the number of gibbs iterations to converge
    lastckpt = max([int(f[:-5]) for f in os.listdir(ckptdir)])
    numCkpts = lastckpt - convergence + 1

    _, occurrences, doc_count, _ = utils.loadCheckpoint(ckptdir / (str(convergence) + ".ckpt"))
    occurrences = occurrences/numCkpts
    doc_count = doc_count/numCkpts

    for i in range(convergence+1, lastckpt + 1):
        _, occur, doc_c, _ = utils.loadCheckpoint(ckptdir/(str(i) + ".ckpt"))
        occurrences += occur/numCkpts
        doc_count += doc_c/numCkpts
    return doc_count, occurrences

def getProbDist(w, theta, beta, hyperparams):
    α = hyperparams["α"]
    γ = hyperparams["γ"]
    word_count = np.sum(beta, axis=1)

    logp = []
    for k in range(5):
        prob = np.log(γ + beta[k]) - np.log(γ*512 + word_count[k])
        score = np.log(α + theta[k])
        for word in w:
            score += prob[word]
        logp.append(score)
    p = np.exp(logp - np.max(logp))
    p = p/sum(p)
    return p

def getLabels(w, theta, beta, hyperparams):
    p = getProbDist(w, theta, beta, hyperparams)
    # Assign this document to a new class, chosen randomly, and add back the counts
    k = np.random.choice(5, p=p)
    return k


def perplexity(w, theta, beta, hyperparams):
    α = hyperparams["α"]
    γ = hyperparams["γ"]
    word_count = np.sum(beta, axis=1)

    logp = []
    for k in range(5):
        prob = np.log(γ + beta[k]) - np.log(γ*512 + word_count[k])
        score = np.log(α + theta[k]) - np.log(5*α + sum(theta))
        for word in w:
            score += prob[word]
        logp.append(score)
    p = sum((mp.matrix(logp)).apply(mp.exp))

    return mp.power(2,(-1*mp.log(p, 2)/len(w)))


def testPerplexity(documents, theta, beta, hyperparams):
    pp = 0
    for i,d in enumerate(documents):
        if (i + 1) % int(len(documents)/100) == 0:
            print(str((i + 1) *100 / float(len(documents))) + "% completed!")
        w = utils.loadWordsF(d, hyperparams["activation_level"])
        pp += perplexity(w, theta, beta, hyperparams)/len(documents)
    return pp

def confusionMatrix(documents, theta, beta, hyperparams):
    labelled = {}
    for i, d in enumerate(documents):
        if (i + 1) % int(len(documents)/100) == 0:
            print(str((i + 1) *100 / float(len(documents))) + "% completed!")
        labelled[d] = getLabels(utils.loadWordsF(d, hyperparams["activation_level"]), theta, beta, hyperparams)
    print("Finished labelling files")

    groundtruth = utils.groundtruth()
    for group, members in groundtruth.items():
        distribution = [0]*5
        for d in members:
            if d in labelled:
                distribution[labelled[d]] += 1
        print(group, distribution)


hyperparams = {"α": 10.0,
               "γ": 1.0,
               "activation_level": 0.3,
               "fold": 0}

# Load test data
test_dir = pathlib.Path(r"D:\PML\Data\test")
labels = os.listdir(str(test_dir))
test_data_label = [[] for i in range(len(labels))]
for index, sub_dir in enumerate(labels):
    test_data_label[index] = os.listdir(test_dir/sub_dir)
test_data = [j for i in test_data_label for j in i]
print("Data Loaded")

ckptdir = pathlib.Path(r"D:\PML\Data\repmean\checkpoints\a_{0}\g_{1}\act_{2}_{3}".format(hyperparams["α"], hyperparams["γ"], hyperparams["activation_level"], hyperparams["fold"]))
# checkConvergence(ckptdir)

# Construct model using Gibbs Samples (MLE)
theta, beta = getModel(ckptdir, convergence=60)

top5feats = [np.argsort(-1*b)[:5] for b in beta]
print(top5feats)

pp = testPerplexity(test_data, theta, beta, hyperparams)
print(hyperparams)
print("Perplexity Score:", pp)

confusionMatrix(test_data, theta, beta, hyperparams)