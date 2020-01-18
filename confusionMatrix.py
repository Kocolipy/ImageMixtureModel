import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, title):
    cmap = plt.get_cmap('Blues')
    x_names = range(1, 6)
    y_names = ["inside", "outside", "food", "menu", "drink"]

    fig, ax = plt.subplots()
    plt.imshow(cm, cmap=cmap)
    plt.colorbar()

    ax.set_xticks(np.arange(len(x_names)))
    ax.set_xticklabels(x_names)
    ax.set_yticklabels([1] + y_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm[i, j]), fontname="Calibri", fontsize=16,
                 horizontalalignment="center",  va="center",
                 color="white" if cm[i, j] > 50 else "black")

    plt.tight_layout()
    # plt.title(title)
    plt.ylabel('Ground Truth Labels')
    plt.xlabel("Topics")
    plt.show()


groups = np.array([
    [48,21,75,3541,1560],
    [3,5,5,958,183],
    [4480,5882,394,19,713],
    [10,0,3,38,252],
    [80,7,1533,35,158]]
)
classDist = np.array([g/sum(g)*100 for g in groups])
plot_confusion_matrix(classDist, "Percentage of topics in each label")

topicDist = np.array(list(zip(*[l/sum(l)*100 for l in zip(*groups)])))
plot_confusion_matrix(topicDist, "Percentage of labels in each topic")