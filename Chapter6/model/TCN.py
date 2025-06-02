import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import DataLoader

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size=5):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = in_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)




class Classifier_TCN(nn.Module):
    def __init__(self, num_classes, in_channels=1, num_channels=6, hiden_dim=128, kernel_size=10,**kwargs):
        super(Classifier_TCN,self).__init__()
        self.tcn = TemporalConvNet(in_channels, [hiden_dim]*num_channels, kernel_size)
        self.fc = nn.Linear(hiden_dim, num_classes)

    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1]  
        return self.fc(x)


def TCN(num_classes=10, **kwargs):
    model = Classifier_TCN(num_classes, **kwargs)
    return model

