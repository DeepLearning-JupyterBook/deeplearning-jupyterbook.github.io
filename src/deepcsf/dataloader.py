"""
This module contains all the related code about the dataset and dataloader.

If this module becomes too completed (too many functions) it can be broken down into smaller modules
e.g., specialised for:
 dataset reading/generating,
 testing-training datasets,
 stimuli manipulation functions such as image processing.
"""

import numpy as np
import random

import cv2

from torch.utils import data as torch_data
import torchvision.transforms as torch_transforms

labels_map = {
    0: "circle",
    1: "ellipse",
    2: "rectangle"
}


def create_random_shape(img_size):
    """This function generates geometrical shapes on top of a background image."""
    # choosing a colour for the shape, there is a slim chance of being identical to the background,
    # we can consider this the noise in our dataset!
    img = np.zeros((img_size, img_size, 3), dtype='uint8') + 128

    colour = [random.randint(0, 255) for _ in range(3)]
    point1 = np.random.randint(img.shape[0] // 4, 3 * (img.shape[0] // 4), 2)

    # drawing a random geometrical shape
    shape_ind = np.random.randint(0, len(labels_map))
    # when the thickness is negative, the shape is drawn filled
    thickness = -1

    if shape_ind == 0:  # circle
        radius = np.random.randint(10, img.shape[0] // 4)
        img = cv2.circle(img, point1, radius, color=colour, thickness=thickness)
    elif shape_ind == 1:  # ellipse
        axes = [
            np.random.randint(10, 20),
            np.random.randint(30, img.shape[0] // 4)
        ]
        angle = np.random.randint(0, 360)
        img = cv2.ellipse(img, point1, axes, angle, 0, 360, color=colour, thickness=thickness)
    else:  # rectangle
        point2 = np.random.randint(0, img.shape[0], 2)
        img = cv2.rectangle(img, point1, point2, color=colour, thickness=thickness)
    return img, shape_ind


def _adjust_contrast(image, amount):
    return (1 - amount) / 2.0 + np.multiply(image, amount)


class ContrastDataset(torch_data.Dataset):
    def __init__(self, num_imgs, target_size, transform=None):
        """
        Parameters:
        ----------
        num_imgs : int
            The number of samples in the dataset.
        target_size : int
            The spatial resolution of generated images.
        transform : List, optional
            The list of transformation functions,
        """
        self.num_imgs = num_imgs
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, _idx):
        # our routine doesn't need the idx, which is the sample number
        img1, _ = create_random_shape(self.target_size)
        # we make two images, one with high-contrast and another with low-contrast
        img1 = img1.astype('float32') / 255
        img2 = img1.copy()

        rnd_contrasts = np.random.uniform(0.04, 1, 2)
        # network's task is to find which image has a higher contrast
        gt = np.argmax(rnd_contrasts)

        img1 = _adjust_contrast(img1, rnd_contrasts[0])
        img2 = _adjust_contrast(img2, rnd_contrasts[1])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1.float(), img2.float(), gt


def sinusoid_grating(img_size, amp, omega, rho, lambda_wave):
    if type(img_size) not in [list, tuple]:
        img_size = (img_size, img_size)
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(img_size[0] / 2.0), int(img_size[1] / 2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))

    stimuli = amp * np.cos((omega[0] * x + omega[1] * y) / lambda_wave + rho)
    return stimuli


class GratingsDataset(torch_data.Dataset):
    def __init__(self, target_size, contrasts, sf, transform=None):
        """
        Parameters:
        ----------
        target_size : int
            The spatial resolution of generated images.
        contrasts : List
            A list of two elements corresponding to the grating contrast.
        sf : int
            The spatial frequency of grating.
        transform : List, optional
            The list of transformation functions,
        """
        self.target_size = target_size
        self.transform = transform
        self.contrasts = contrasts
        self.sf = sf
        self.thetas = np.arange(0, np.pi + 1e-3, np.pi / 12)

    def __len__(self):
        return len(self.thetas)

    def __getitem__(self, idx):
        theta = self.thetas[idx]
        omega = [np.cos(theta), np.sin(theta)]
        lambda_wave = (self.target_size * 0.5) / (np.pi * self.sf)
        # generating the gratings
        sinusoid_param = {
            'amp': self.contrasts[0], 'omega': omega, 'rho': 0,
            'img_size': self.target_size, 'lambda_wave': lambda_wave
        }
        img1 = sinusoid_grating(**sinusoid_param)
        sinusoid_param['amp'] = self.contrasts[1]
        img2 = sinusoid_grating(**sinusoid_param)

        # if the target size is even, the generated stimuli is 1 pixel larger.
        if np.mod(self.target_size, 2) == 0:
            img1 = img1[:-1, :-1]
            img2 = img2[:-1, :-1]

        # multiply by a Gaussian
        radius = int(self.target_size / 2.0)
        [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))

        sigma = self.target_size / 6
        gauss_img = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

        if np.mod(self.target_size, 2) == 0:
            gauss_img = gauss_img[:-1, :-1]
        gauss_img = gauss_img / np.max(gauss_img)

        img1 *= gauss_img
        img2 *= gauss_img

        # bringing the image in the range of 0-1
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2

        # converting it to 3 channel
        img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
        img2 = np.repeat(img2[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        gt = np.argmax(self.contrasts)
        return img1.float(), img2.float(), gt


def train_dataloader(args):
    transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean=args.mean, std=args.std)
    ])
    train_db = ContrastDataset(args.train_samples, args.target_size, transform)

    train_loader = torch_data.DataLoader(
        train_db, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, sampler=None
    )
    return train_loader


def test_dataloader(args, sf, contrast):
    transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean=args.mean, std=args.std)
    ])
    # batch size is hardcoded to 13 because our test dataset contains 13 samples
    test_loader = torch_data.DataLoader(
        GratingsDataset(args.target_size, transform=transform, contrasts=[0, contrast], sf=sf),
        batch_size=13, shuffle=False, num_workers=0, pin_memory=True, sampler=None
    )
    return test_loader
