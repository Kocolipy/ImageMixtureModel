import torch.nn as nn
import torchvision

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def loadVGG():
    vgg = torchvision.models.vgg19_bn(pretrained=True);
    for param in vgg.features.parameters():
        param.require_grad = False

    # Remove classifier layer
    vgg.classifier = Identity()

    return vgg