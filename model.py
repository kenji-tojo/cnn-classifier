import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.feature = _create_feature_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = _create_fc_layers()
    
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

def _create_feature_layers():
    skeleton = [16, 16, -1, 32, 32, -1]
    layers = list()
    in_channels = 1
    for out_channels in skeleton:
        if out_channels < 0:
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(1,1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers)

def _create_fc_layers():
    return nn.Sequential(
        nn.Linear(7*7*32, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 10))