import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from model import weights_init
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, Generator, Discriminator, learning_rate, noise_dim):
        # Device, noise_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim

        # Init Generator
        self.generator = Generator.to(self.device)
        weights_init(self.generator)

        # Init Discriminator
        self.discriminator = Discriminator.to(self.device)
        weights_init(self.discriminator)

        # Optimizer
        self.gen_opt = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()

        # Tensorboard writer
        self.fixed_noise = torch.randn(32, noise_dim, 1, 1).to(self.device)
        self.writer_real = SummaryWriter(f"logs/real")
        self.writer_fake = SummaryWriter(f"logs/fake")
    
    def train(self,dataloader, epochs):
        step = 0
        self.generator.train()
        self.discriminator.train()

        for epoch in range(1, epochs + 1):
            print('-' * 59)
            print(f"Epoch [{epoch}/{epochs}]")
            print('-' * 59)
            for batch_idx, (real, _) in enumerate(dataloader):
                batch_size = real.size(0)

                # Inference
                real = real.to(self.device)
                noise = torch.rand(batch_size, self.noise_dim, 1 ,1)
                fake = self.generator(noise)

                # Train Discriminator
                disc_real = self.discriminator(real).reshape(-1)
                loss_disc_real = self.criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = self.discriminator(fake.detach().reshape(-1))
                loss_disc_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                discriminator_loss = 0.5 * (loss_disc_real + loss_disc_fake)
                self.discriminator.zero_grad()
                discriminator_loss.backward()
                self.dis_opt.step()

                # Train Generator
                output = self.discriminator(fake).reshape(-1)
                generator_loss = self.criterion(output, torch.ones_like(output))
                self.generator.zero_grad()
                generator_loss.backward()
                self.gen_opt.step()

                # Print loss occasionally 
                if batch_idx % 100 == 0:
                    print(f"\tBatch [{batch_idx}/{len(dataloader)}]: Discriminator Loss {discriminator_loss} - Generator Loss {generator_loss}")
                
                with torch.no_grad():
                    fake = self.generator(self.fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    self.writer_real.add_image("Real", img_grid_real, global_step=step)
                    self.writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
