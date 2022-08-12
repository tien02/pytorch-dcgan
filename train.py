import argparse 
from config import config

import torch
from torchvision.datasets import ImageFolder
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.model import Generator, Discriminator
from src.trainer import Trainer

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None, help="Path to dataset, default is MNIST dataset")
    opt = parser.parse_args()
    return opt

def run(datapath=None):
    # transform
    trans = transforms.Compose(
    [
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(config.CHANNELS_IMG)], [0.5 for _ in range(config.CHANNELS_IMG)]
        ),
    ]
    )

    # dataset
    if datapath is None:
        dataset = datasets.MNIST(root="dataset/", train=True, transform=trans, download=True)
    else:
        dataset = ImageFolder(datapath, transform=trans)

    # dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Declare Generator
    gen = Generator(config.NOISE_DIM, config.CHANNELS_IMG, config.FEATURES_GEN)

    # Declare Discriminator
    disc = Discriminator(config.CHANNELS_IMG, config.FEATURES_DISC)

    # Declare Trainer
    trainer = Trainer(gen, disc, config.LEARNING_RATE, config.NOISE_DIM, "./weight")

    # Train
    trainer.train(dataloader, config.NUM_EPOCHS)

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)