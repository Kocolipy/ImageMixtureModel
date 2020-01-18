import pathlib
import os
import mpmath as mp
import numpy as np
import json
import utils
import matplotlib.pyplot as plt

def checkConvergence(ckptdir):
    lastckpt = max([int(f[:-5]) for f in os.listdir(ckptdir)])

    _, _, doc_count, _ = utils.loadCheckpoint(ckptdir/"0.ckpt")
    doc_total = sum(doc_count)
    docs = [[d] for d in doc_count/doc_total]

    for i in range(1, lastckpt + 1):
        _, _, doc_count, _ = utils.loadCheckpoint(ckptdir/(str(i) + ".ckpt"))
        for i, d in enumerate(doc_count/doc_total):
            docs[i].append(d)

    x = np.arange(lastckpt+1) + 1
    for k in range(len(docs)):
        plt.plot(x, docs[k])
    plt.title("Distributions of each topics")
    plt.xlabel("Number of Gibbs sweeps")
    plt.ylabel("Probability")
    plt.show()


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


def getLabels(w, theta, beta, hyperparams):
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


def valPerplexity(val_dir, theta, beta, hyperparams):
    documents = utils.loadDocuments(directory=val_dir)
    pp = 0
    for i,d in enumerate(documents):
        if (i + 1) % int(len(documents)/100) == 0:
            print(str((i + 1) *100 / float(len(documents))) + "% completed!")
        w = utils.loadWordsF(d, hyperparams["activation_level"])
        pp += perplexity(w, theta, beta, hyperparams)/len(documents)
    return pp

# def confusionMatrix(test_dir, theta, beta, hyperparams):
#     α = 10
#     γ = 0.1
#     activation_level = 1.0
#     documents = utils.loadDocuments(test_dir)
#
#     labelled = {}
#     for i, d in enumerate(documents):
#         if (i + 1) % int(len(documents)/100) == 0:
#             print(str((i + 1) *100 / float(len(documents))) + "% completed!")
#         labelled[d] = getLabels(utils.loadWordsF(d, activation_level), ckptfile, α=α, γ=γ)
#     print("Finished labelling files")
#
#     groundtruth = utils.groundtruth()
#     for group, members in groundtruth.items():
#         distribution = [0]*5
#         for d in members:
#             if d in labelled:
#                 distribution[labelled[d]] += 1
#         print(group, np.array(distribution)/sum(distribution)*100)


hyperparams = {"α": 10.0,
               "γ": 1.0,
               "activation_level":  0.3,
               "fold":0}

val_dir = pathlib.Path("D:\PML\Data\\folds\\{0}\\val".format(hyperparams["fold"]))
ckptdir = pathlib.Path("D:\PML\Data\\repmean\\checkpoints\\a_{0}\\g_{1}\\act_{2}_0".format(hyperparams["α"], hyperparams["γ"], hyperparams["activation_level"]))

checkConvergence(ckptdir)

# # Construct model using Gibbs Samples (MLE)
theta, beta = getModel(ckptdir, convergence=60)

# top5feats = [np.argsort(-1*b)[:5] for b in beta]
# print(top5feats)

pp = valPerplexity(val_dir, theta, beta, hyperparams)
print(hyperparams)
print("Perplexity Score:", pp)
print(sorted(list(np.array(theta)/sum(theta)*100)))
