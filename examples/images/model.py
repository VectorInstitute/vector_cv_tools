import torch
import torch.nn as nn
NC = 3
NDF = 128
NGF = 128
NZ = 100


class Decoder(nn.Module):
    """
    The model architecture is taken from https://github.com/pytorch/examples/issues/70
    """

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(NZ, NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            # state size. (NGF*16) x 4 x 4
            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # state size. (NGF*8) x 8 x 8
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # state size. (NGF*4) x 16 x 16
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # state size. (NGF*2) x 32 x 32
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # state size. (NGF) x 64 x 64
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (NC) x 128 x 128
        )

    def forward(self, x):
        return self.main(x)


class Encoder(nn.Module):
    """
    The model architecture is taken from https://github.com/pytorch/examples/issues/70
    """

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is (NC) x 128 x 128
            nn.Conv2d(NC, NDF, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF) x 64 x 64
            nn.Conv2d(NDF, NDF * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*2) x 32 x 32
            nn.Conv2d(NDF * 2, NDF * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*4) x 16 x 16
            nn.Conv2d(NDF * 4, NDF * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x 8 x 8
            nn.Conv2d(NDF * 8, NDF * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*16) x 4 x 4
            nn.Conv2d(NDF * 16, 2 * NZ, 4, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)


def loss_fn(x, recon_batch, mu, logvar):
    """Function taken and modified from
        https://github.com/pytorch/examples/tree/master/vae
    """
    B = x.size(0)
    MSE = (x - recon_batch).pow(2).sum() / B
    KLD = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return MSE + KLD


class ConvVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_out = self.encoder(x)
        mu, logvar = enc_out[..., :100], enc_out[..., 100:]
        z = self.reparameterize(mu, logvar)
        recon_batch = self.decoder(z.unsqueeze(-1).unsqueeze(-1))
        return recon_batch, mu, logvar
