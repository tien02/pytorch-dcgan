import torch 
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d * 2),
            self._block(feature_d * 2, feature_d * 4),
            self._block(feature_d * 4, feature_d * 8),
            nn.Conv2d(feature_d * 8, 1, 4, 2, 0),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.cnn(input)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels, feature_g):
        super(Generator, self).__init__()
        self.cnn = nn.Sequential(
            self._block(noise_channels, feature_g * 16, stride=1, padding=0),
            self._block(feature_g * 16, feature_g * 8),
            self._block(feature_g * 8 , feature_g * 4),
            self._block(feature_g * 4, feature_g * 2),
            nn.ConvTranspose2d(feature_g * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.cnn(input)

    def _block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() 
        )

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    print("Test model.py.....", end=".")
    batch, channels, imgH, imgW = 4, 3, 64, 64
    noise_dim = 100
    img = torch.rand(batch, channels, imgH, imgW)
    noise = torch.rand(batch, noise_dim, 1, 1)
    disc = Discriminator(channels, 64)
    gen = Generator(noise_dim, channels, 64)
    assert disc(img).size() == (batch, 1, 1, 1), "ERROR: Discriminator Fail!" 
    assert gen(noise).size() == (batch, channels, imgH, imgW), "ERROR: Generator Fail!"
    print("Success!")

if __name__ == '__main__':
    test()