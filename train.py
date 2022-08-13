import argparse 
from . import config

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import Generator, Discriminator, initialize_weights

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=None, help="Path to dataset, default is MNIST dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, help="Save checkpoint")
    opt = parser.parse_args()
    return opt

def run(datapath=None, checkpoint=None, save=False):
    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # transform
    trans = transforms.Compose(
    [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
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

    # Declare Generator & Discriminator
    gen = Generator(config.NOISE_DIM, config.CHANNELS_IMG, config.FEATURES_GEN).to(device)
    disc = Discriminator(config.CHANNELS_IMG, config.FEATURES_DISC).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5,0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEANRING_RATE, betas=(0.5,0.999))

    if checkpoint is None:
        start = 1
        end = config.NUM_EPOCHS
       # Initialize weight
        initialize_weights(gen)
        initialize_weights(disc)
    else:
        checkpoint = torch.load(checkpoint, map_location=device)
        start = checkpoint['epoch'] + 1
        end = checkpoint['epoch'] + config.NUM_EPOCHS

        gen.load_state_dict(checkpoint['gen_state'])
        disc.load_state_dict(checkpoint['disc_state'])

        opt_gen.load_state_dict(checkpoint['gen_opt'])
        opt_disc.load_state_dict(checkpoint['disc_opt'])
    
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, config.NOISE_DIM,1,1).to(device)
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(start, end + 1):
        print('-' * 59)
        print(f"Epoch [{epoch}/{end}]")
        print()
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn((config.BATCH_SIZE, config.NOISE_DIM, 1, 1)).to(device)
            fake = gen(noise)

            ### Train Discriminator
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = 0.5 * (loss_disc_real + loss_disc_fake)
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            ### Print
            if batch_idx % 100 == 0:
                print("Batch [{}/{}]: Loss D {:.2f} - Loss G {:.2f}".format(batch_idx, len(dataloader), loss_disc, loss_gen))
            
            with torch.no_grad():
                fake = gen(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1
    # Checkpoint
    if save:
        if not os.path.exists("weight"):
            os.mkdir("./weight") 
        torch.save({
            'epoch': epoch,
            'gen_state': gen.state_dict(),
            'disc_state': disc.state_dict(),
            'gen_opt': opt_gen.state_dict(),
            'disc_opt': opt_disc.state_dict()
        }, 'weight/{}.pt'.format(torch.randint(0, 10, (1,1)).item()))

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)