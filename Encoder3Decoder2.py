import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.actv1 = nn.PReLU(num_parameters=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.actv2 = nn.PReLU(num_parameters=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, signal):
        x = self.conv1(signal)
        x = self.bn1(x)
        x = self.actv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actv2(x)
        x = self.pool2(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.trConv1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.actv1 = nn.PReLU(num_parameters=1)
        self.trConv2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        self.trConv3 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.actv2 = nn.PReLU(num_parameters=1)
        self.trConv4 = nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self,features):
        x = self.trConv1(features)
        x = self.bn1(x)
        x = self.actv1(x)
        x = self.trConv2(x)
        x = self.trConv3(x)
        x = self.bn2(x)
        x = self.actv2(x)
        signal = self.trConv4(x)

        return signal


class Encoder3Decoder(nn.Module):
    def __init__(self):
        super(Encoder3Decoder, self).__init__()
        self.encoder = Encoder()
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.decoder3 = Decoder()

    def forward(self, signal):
        signal = signal.unsqueeze(1)
        x = self.encoder(signal)
        sig1 = self.decoder1(x)
        sig2 = self.decoder2(x)
        sig3 = self.decoder3(x)
        res = torch.stack((sig1,sig2,sig3),dim=1)
        res = res.squeeze(2)
        return res
    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


if __name__ == '__main__':
    model = Encoder3Decoder()
    data = torch.rand(10,4000)
    source = model(data)
    print(source)
    # print(sig2.shape)
    # print(sig3.shape)