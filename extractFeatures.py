from __future__ import print_function, division

import torch
import pandas as pd
from torchvision import datasets, transforms
import os
import pathlib
import shutil
import vgg


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class CNNCollector():
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def save(self, outputs, paths):
        outputs = outputs.tolist()
        images = [image.split('\\')[-1] for image in paths]
        for i in range(len(images)):
            dataFrame = pd.DataFrame(data={"output": outputs[i]})
            dataFrame.to_feather(str(self.directory / (images[i] + ".ftr")))


def loadData(datadir, batch_size=10, num_workers=10):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])
    traindir = pathlib.Path(datadir) / "photos"
    train_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(traindir, trans),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    testdir = pathlib.Path(datadir) / "test"

    test_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, trans),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Get pre-trained model
    model = vgg.loadVGG()
    if use_cuda:
        model.cuda()  # .cuda() will move everything to the GPU side

    train_loader, test_loader = loadData("D:\PML\Data")

    out_dir = pathlib.Path("D:\PML\\Data\\features")
    collector = CNNCollector(out_dir)
    for i, data in enumerate(train_loader):
        inputs, _, paths = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        collector.save(outputs, paths)
    print("Features extracted from Training Set")
    torch.cuda.empty_cache()

    for i, data in enumerate(test_loader):
        inputs, _, paths = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        collector.save(outputs, paths)
    print("Features extracted from Test Set")
