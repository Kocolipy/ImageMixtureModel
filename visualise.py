import argparse
import torch
import numpy as np
import torchvision
import os
import pathlib
import matplotlib.pyplot as plt
import vgg
import cv2

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def crop(img):
    (x,y,z) = img.shape
    return img[int(x/20):int(x*19/20), int(y/20):int(y*19/20), :]


class FilterVisualizer():
    def __init__(self, size, upscaling_steps, upscaling_factor, epoch):
        self.size, self.upscaling_steps, self.upscaling_factor, self.epoch = size, upscaling_steps, upscaling_factor, epoch
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.criterion = torch.nn.MSELoss()

        # Load VGG model and freeze weights
        self.model = vgg.loadVGG()
        self.model.eval()
        self.model.to(self.device)
        print("VGG model loaded ...")

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
        self.unnormalize = UnNormalize(torch.from_numpy(np.array([0.485, 0.456, 0.406])), torch.from_numpy(np.array([0.229, 0.224, 0.225])))

    def visualize(self, filter, lr=0.1, blur=None):
        sz = self.size

        while True:
            # Generate a random image which produce a value in the selected filter
            img = np.uint8(np.random.uniform(128, 200, (3, sz, sz))) / 255  # generate random image
            img_var = self.normalize(torch.tensor(img)).unsqueeze(0).float().requires_grad_()
            img_var = img_var.to(self.device)

            x = self.model.features(img_var)
            output = self.model.avgpool(x)
            loss = -output[0, filter].mean()

            if loss < 0:
                print("Free! Beginning optimisation!")
                break

            print('Stuck in minimum. Initialising new starting image')

        for i in range(self.upscaling_steps):
            # convert image to Variable that requires grad
            img_var = self.normalize(torch.tensor(img)).unsqueeze(0).float()
            img_var = img_var.to(self.device)
            img_var.requires_grad_()

            optimizer = torch.optim.Adam([img_var.requires_grad_()], lr=lr, weight_decay=1e-6)
            for n in range(self.epoch):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                x = self.model.features(img_var)
                output = self.model.avgpool(x)
                loss = -output[0, filter].mean()
                loss.backward()
                optimizer.step()

            for n in range(int(self.epoch / 2)):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                x = self.model.features(img_var)
                output = self.model.avgpool(x)
                output = torch.mean(torch.mean(output, 3),2)
                goal = torch.zeros(1, 512)
                goal[0, filter] = output[0, filter]*1.5
                goal = goal.to(self.device)
                loss = self.criterion(output, goal)
                loss.backward()
                optimizer.step()
            print("Activation of filter {0}:".format(filter), loss.item())

            # Denormalise and move to CPU for processing
            img_var = self.unnormalize(img_var.squeeze(0)).cpu()
            img = img_var.data.numpy().transpose(1, 2, 0)
            sz = int(self.upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)  # scale image up
            if blur is not None:
              img = cv2.blur(img,(blur, blur))  # blur image to reduce high frequency patterns
            img = img.transpose(2,0,1)
        self.output = img.transpose(1,2,0)
        self.save(filter)

    def save(self, filter):
        path = pathlib.Path(os.getcwd()) / str(filter)
        path.mkdir(parents=True, exist_ok=True)
        img = crop(self.output)
        plt.imsave(str(path / "{0}_{1}_{2}_{3}.jpg"
                    .format(self.size, self.epoch, self.upscaling_steps, self.upscaling_factor)),
                   np.clip(img, 0, 1))


parser = argparse.ArgumentParser(description='Visualise VGG')
parser.add_argument('filter', metavar='filter', type=int, nargs='+',
                    help='Filter layer to visualise')
parser.add_argument('--size', dest='size', type=int, nargs='?',
                    help='Image size to start  (Default: 200)')
parser.add_argument('--epochs', dest='epochs', type=int, nargs='?',
                    help='Number of epochs to run for  (Default: 20)')
parser.add_argument('--upscaling', dest='upscaling_steps', type=int, nargs='?',
                    help='Number of upscaling to perform  (Default: 12)')
parser.add_argument('--scaling_f', dest='upscaling_factor', type=float, nargs='?',
                    help='Scaling factor for upscaling  (Default: 1.2)')

args = parser.parse_args()
params = {"filter": args.filter[0],
           "size": args.size if args.size else 200,
           "epochs": args.epochs if args.epochs else 100,
           "upscaling_steps": args.upscaling_steps if args.upscaling_steps else 12,
           "upscaling_factor": args.upscaling_factor if args.upscaling_factor else 1.2}

FV = FilterVisualizer(size=params["size"], upscaling_steps=params["upscaling_steps"], upscaling_factor=params["upscaling_factor"],  epoch=params["epochs"])
FV.visualize(params["filter"], blur=5)
