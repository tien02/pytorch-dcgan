import torch
import torch.nn as nn
import torch.optim as optim



def Trainer(dataloader, Generator, Discriminator, Gen_opt, Dis_opt, criterion, device):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.rand()
