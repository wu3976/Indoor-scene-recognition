import torch
import torch.nn as nn


class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvUnit, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input_batch):
        return self.model(input_batch)


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualUnit, self).__init__()
        self.prep = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.mainway = ConvUnit(in_channels, out_channels, stride=stride)
        self.shortcut = lambda x: x
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, padding=0, stride=stride)

    def forward(self, input_batch):
        data_in = self.prep(input_batch)
        main_out = self.mainway(data_in)
        shortcut_out = self.shortcut(data_in)

        return main_out + shortcut_out


class ResidualGroup(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualGroup, self).__init__()
        self.model = nn.Sequential(
            ResidualUnit(in_channels, out_channels, stride=stride),
            ResidualUnit(out_channels, out_channels),
            ResidualUnit(out_channels, out_channels)
        )

    def forward(self, input_batch):
        return self.model(input_batch)


class LinearUnit(nn.Module):

    def __init__(self, in_features, out_features):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_features)
        )

    def forward(self, input_batch):
        return self.model(input_batch)


class Flatten(nn.Module):

    def __init__(self, dim):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, input_batch):
        return input_batch.view(input_batch.size(0), -1)


class WideResnet(nn.Module):

    def __init__(self):
        super(WideResnet, self).__init__()
        self.resgroup1 = ResidualGroup(16, 96, 2)  # 96x128x128
        self.resgroup3 = ResidualGroup(96, 384, 2)  # 384x64x64
        # self.resgroup4 = ResidualGroup(384, 768, 2)  # 768x32x32
        # self.resgroup5 = ResidualGroup(768, 1536, 2)  # 1536x16x16
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3),
            self.resgroup1,
            self.resgroup3,
            nn.AdaptiveAvgPool2d((1, 1)),  # 384x1x1
            Flatten(dim=1),  # bsx384
            LinearUnit(384, 67)
        )

    def forward(self, input_batch):
        return self.model(input_batch)

    def accuracy(self, prediction, target):
        _, max_pos = torch.max(prediction, dim=1)
        return torch.tensor(torch.sum(max_pos == target).item() / len(max_pos))

    def train_step(self, input_batch, target_batch):
        prediction = self(input_batch)
        return nn.functional.cross_entropy(prediction, target_batch)

    def validation_step(self, data_batch, target_batch):
        prediction = self(data_batch)
        accuracy = self.accuracy(prediction, target_batch)
        loss = nn.functional.cross_entropy(prediction, target_batch)
        return {"loss": loss, "accuracy": accuracy}