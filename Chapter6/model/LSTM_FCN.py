import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Classifier_LSTMFCN(nn.Module):

    def __init__(self, num_classes, seq_length, num_cells=64, **kwargs):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=seq_length,
            hidden_size=num_cells,
            batch_first=True,
            dropout=0.8,
        )

        # Convolutional block
        self.conv1 = nn.Conv1d(
            1, 128, kernel_size=8, padding="same", bias=True
        )
        init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding="same", bias=True)
        init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same", bias=True)
        init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Final dense layer
        self.fc = nn.Linear(num_cells + 128, num_classes)

    def forward(self, x):
        # LSTM part
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm.squeeze(1)

        # Convolutional part
        # Permute x to make it (batch_size, channels, sequence_length)
        x_conv = F.relu(self.bn1(self.conv1(x)))
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = F.relu(self.bn3(self.conv3(x_conv)))
        x_conv = torch.flatten(self.global_avg_pool(x_conv), 1)  # Flatten the output

        # Concatenate LSTM and CNN features
        x = torch.cat((x_lstm, x_conv), dim=1)

        # Final dense layer with softmax
        x = self.fc(x)

        # return F.softmax(x, dim=1)
        return x
    

def LSTMFCN(num_classes=10, seq_length=100, **kwargs):
    model = Classifier_LSTMFCN(num_classes, seq_length,**kwargs)
    return model