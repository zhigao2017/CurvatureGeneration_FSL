import torch
import torch.nn as nn

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        z_dim=64

        self.conv1=nn.Conv2d(x_dim, hid_dim, 3, padding=1)
        self.bn1=nn.BatchNorm2d(hid_dim)
        self.conv2=nn.Conv2d(hid_dim, hid_dim, 3, padding=1)
        self.conv3=nn.Conv2d(hid_dim, hid_dim, 3, padding=1)
        self.conv4=nn.Conv2d(hid_dim, z_dim, 3, padding=1)


    def forward(self, x):

        x = self.conv1(x)
        print('conv1_x',torch.sum(x))
        x = self.conv2(x)
        print('conv2_x',torch.sum(x))
        x = self.conv3(x)
        print('conv3_x',torch.sum(x))
        x = self.conv4(x)
        print('conv4_x',torch.sum(x))
        #x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        return x

